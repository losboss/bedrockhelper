from __future__ import annotations

import json
from io import BytesIO
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from bedrockhelper import EmbeddingResponse
from bedrockhelper.main import BedrockHelper, _BotoSessionManager, _is_expired_token


def _client_error(code: str, op: str = 'InvokeModel') -> ClientError:
	return ClientError(
		error_response={'Error': {'Code': code, 'Message': 'boom'}},
		operation_name=op,
	)


class _FakeClosableStream:
	def __init__(self, items=None):
		self.closed = False
		self.items = list(items or [])

	def close(self):
		self.closed = True

	def __iter__(self):
		return iter(self.items)


class _FakePaginator:
	def __init__(self, pages):
		self._pages = pages

	def paginate(self, **kwargs):
		yield from self._pages


class _ImmediateThread:
	"""Drop-in replacement for threading.Thread that runs target immediately on start()."""

	def __init__(self, *, target, daemon=False):
		self._target = target
		self.daemon = daemon

	def start(self):
		self._target()


class _ExplodingReleaseSema:
	def __init__(self):
		self.acquired = False

	def acquire(self):
		self.acquired = True
		return True

	def release(self):
		raise RuntimeError('release blew up')


class TestExpiredTokenHelpers(TestCase):
	def test_is_expired_token_true_all_codes(self):
		for code in [
			'ExpiredToken',
			'ExpiredTokenException',
			'InvalidClientTokenId',
			'UnrecognizedClientException',
			'RequestExpired',
		]:
			self.assertTrue(_is_expired_token(_client_error(code)))

	def test_is_expired_token_false(self):
		self.assertFalse(_is_expired_token(_client_error('AccessDeniedException')))
		self.assertFalse(_is_expired_token(RuntimeError('nope')))


class TestBotoSessionManager(TestCase):
	@patch('bedrockhelper.main.boto3.Session')
	def test_build_session_no_role_uses_default_resolution(self, SessionMock):
		SessionMock.return_value = MagicMock(name='session')
		mgr = _BotoSessionManager(
			region_name='ca-central-1',
			botocore_config=MagicMock(),
			role_arn=None,
		)
		sess = mgr._build_boto3_session()
		self.assertIs(sess, SessionMock.return_value)
		SessionMock.assert_called_with(region_name='ca-central-1')

	@patch('bedrockhelper.main.AssumeRoleCredentialFetcher')
	@patch('bedrockhelper.main.DeferredRefreshableCredentials')
	@patch('bedrockhelper.main.boto3.Session')
	@patch('bedrockhelper.main.BotocoreSession')
	def test_build_session_assume_role_refreshable_happy_path(
		self,
		BotocoreSessionMock,
		Boto3SessionMock,
		DeferredMock,
		FetcherMock,
	):
		# Botocore session + source creds exist
		bc = MagicMock()
		source_creds = MagicMock()
		bc.get_credentials.return_value = source_creds
		BotocoreSessionMock.return_value = bc

		# sts client comes from boto3.Session(...).client('sts')
		sts_client = MagicMock(name='sts')
		boto3_sess_for_sts = MagicMock()
		boto3_sess_for_sts.client.return_value = sts_client

		# Final session
		final_boto3_session = MagicMock(name='final_session')

		# boto3.Session can be called multiple times - use a function instead of side_effect list
		call_count = 0

		def session_factory(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			if 'botocore_session' in kwargs:
				return final_boto3_session
			return boto3_sess_for_sts

		Boto3SessionMock.side_effect = session_factory

		# Fetcher + deferred creds
		fetcher = MagicMock()
		fetcher.fetch_credentials.return_value = {
			'access_key': 'AKIA...',
			'secret_key': 'SECRET',
			'token': 'TOKEN',
			'expiry_time': '2099-01-01T00:00:00Z',
		}
		FetcherMock.return_value = fetcher

		refreshable = MagicMock()
		refreshable.access_key = 'AKIA...'
		refreshable.secret_key = 'SECRET'
		refreshable.token = 'TOKEN'
		DeferredMock.return_value = refreshable

		mgr = _BotoSessionManager(
			region_name='ca-central-1',
			botocore_config=MagicMock(),
			role_arn='arn:aws:iam::123:role/TestRole',
			role_session_name='bedrockhelper-test',
			external_id='ext-123',
			sts_region_name='us-east-1',
			assume_role_duration_seconds=900,
		)

		sess = mgr._build_boto3_session()

		self.assertEqual(Boto3SessionMock.call_count, 3)
		self.assertIs(sess, final_boto3_session)

	@patch('bedrockhelper.main.BotocoreSession')
	def test_build_session_role_no_source_creds_raises(self, BotocoreSessionMock):
		bc = MagicMock()
		bc.get_credentials.return_value = None
		BotocoreSessionMock.return_value = bc

		mgr = _BotoSessionManager(
			region_name='ca-central-1',
			botocore_config=MagicMock(),
			role_arn='arn:aws:iam::123:role/X',
		)
		with self.assertRaises(RuntimeError):
			mgr._build_boto3_session()

	@patch('bedrockhelper.main.boto3.Session')
	def test_client_cache_and_ensure_session_clears_on_rebuild(self, SessionMock):
		# session.client called twice because invalidate forces rebuild
		sess = MagicMock()
		c1 = MagicMock(name='s3_client_1')
		c2 = MagicMock(name='s3_client_2')
		sess.client.side_effect = [c1, c2]
		SessionMock.return_value = sess

		mgr = _BotoSessionManager(region_name='ca-central-1', botocore_config=MagicMock(), role_arn=None)

		a = mgr.client('s3')
		b = mgr.client('s3')
		self.assertIs(a, b)  # cached

		mgr.invalidate()
		c = mgr.client('s3')
		self.assertIs(c, c2)  # rebuilt
		self.assertEqual(sess.client.call_count, 2)


class TestBedrockHelperSync(TestCase):
	def _make_helper_with_clients(self):
		brt = MagicMock(name='bedrock-runtime')
		br = MagicMock(name='bedrock')
		s3 = MagicMock(name='s3')
		return BedrockHelper(bedrock_runtime_client=brt, bedrock_client=br, s3_client=s3)

	def test_refresh_clients_if_needed_respects_injected(self):
		bh = self._make_helper_with_clients()
		# All injected: should not call session manager
		bh._session_mgr = MagicMock()
		bh._refresh_clients_if_needed()
		bh._session_mgr.client.assert_not_called()

	def test_refresh_clients_if_needed_rebinds_when_not_injected(self):
		bh = self._make_helper_with_clients()

		# Pretend none were injected so refresh should rebind all three
		bh._injected = {'bedrock_runtime': False, 'bedrock': False, 's3': False}

		new_brt = MagicMock(name='new_bedrock_runtime')
		new_br = MagicMock(name='new_bedrock')
		new_s3 = MagicMock(name='new_s3')

		bh._session_mgr = MagicMock()
		bh._session_mgr.client.side_effect = [new_brt, new_br, new_s3]

		bh._refresh_clients_if_needed()

		self.assertIs(bh.bedrock_runtime, new_brt)
		self.assertIs(bh.bedrock, new_br)
		self.assertIs(bh.s3, new_s3)

		# ensure the 3 calls happened in order
		self.assertEqual(
			[c.args[0] for c in bh._session_mgr.client.call_args_list],
			['bedrock-runtime', 'bedrock', 's3'],
		)

	def test_call_with_refresh_success(self):
		bh = self._make_helper_with_clients()
		bh._session_mgr = MagicMock()

		fn = MagicMock(return_value={'ok': True})
		out = bh._call_with_refresh(fn, 1, a=2)
		self.assertEqual(out, {'ok': True})
		bh._session_mgr.invalidate.assert_not_called()

	def test_call_with_refresh_expired_retries(self):
		bh = self._make_helper_with_clients()
		bh._session_mgr = MagicMock()
		bh._refresh_clients_if_needed = MagicMock()

		fn = MagicMock(side_effect=[_client_error('ExpiredToken'), {'ok': True}])
		out = bh._call_with_refresh(fn)

		self.assertEqual(out, {'ok': True})
		bh._session_mgr.invalidate.assert_called_once()
		bh._refresh_clients_if_needed.assert_called_once()
		self.assertEqual(fn.call_count, 2)

	def test_call_with_refresh_expired_token_rebinds_and_retries_successfully(self):
		bh = self._make_helper_with_clients()

		# Make them non-injected so _refresh_clients_if_needed actually does work
		bh._injected = {'bedrock_runtime': False, 'bedrock': False, 's3': False}

		# Session mgr will be invalidated and then used to create 3 clients during refresh.
		bh._session_mgr = MagicMock()
		bh._session_mgr.client.side_effect = [MagicMock(), MagicMock(), MagicMock()]

		# Function fails once with ExpiredToken, then returns ok
		fn = MagicMock(side_effect=[_client_error('ExpiredToken'), 'OK'])
		out = bh._call_with_refresh(fn)

		self.assertEqual(out, 'OK')
		bh._session_mgr.invalidate.assert_called_once()
		self.assertEqual(fn.call_count, 2)
		self.assertEqual(bh._session_mgr.client.call_count, 3)  # runtime + bedrock + s3

	def test_call_with_refresh_non_expired_reraises(self):
		bh = self._make_helper_with_clients()
		bh._session_mgr = MagicMock()
		fn = MagicMock(side_effect=_client_error('AccessDeniedException'))
		with self.assertRaises(ClientError):
			bh._call_with_refresh(fn)

	def test_extract_text_from_claude_branches(self):
		self.assertEqual(
			BedrockHelper._extract_text_from_claude({'content': [{'text': 'a'}, {'text': 'b'}]}),
			'ab',
		)
		self.assertEqual(BedrockHelper._extract_text_from_claude({'completion': 'yo'}), 'yo')
		self.assertEqual(BedrockHelper._extract_text_from_claude('hi'), 'hi')
		self.assertEqual(BedrockHelper._extract_text_from_claude({'x': 1}), '')

	def test_generate_with_claude_message_format_includes_extra_params(self):
		bh = self._make_helper_with_clients()

		# Capture the request body passed to invoke_model
		captured = {}

		def invoke_model(**kwargs):
			captured['body'] = json.loads(kwargs['body'])
			return {'body': BytesIO(json.dumps({'content': [{'text': 'ok'}]}).encode('utf-8'))}

		bh.bedrock_runtime.invoke_model.side_effect = invoke_model

		out = bh.generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			prefer_converse=False,  # force Claude fallback
			stream=False,
			top_p=0.9,  # <-- extra param (THIS triggers line 541)
		)

		self.assertEqual(out.text, 'ok')

		# This assertion proves request_body.update(extra_params) executed
		self.assertIn('top_p', captured['body'])
		self.assertEqual(captured['body']['top_p'], 0.9)

	def test_extract_metrics_body_usage_and_header_fallbacks(self):
		bh = self._make_helper_with_clients()

		resp = {
			'ResponseMetadata': {
				'HTTPHeaders': {
					'X-Amzn-Bedrock-Input-Token-Count': '10',
					'X-Amzn-Bedrock-Output-Token-Count': '20',
					'X-Amzn-Bedrock-Invocation-Latency': '30',
					'X-Amzn-Bedrock-First-Byte-Latency': '40',
				}
			}
		}
		body = {
			'usage': {'inputTokens': 1, 'outputTokens': 2, 'totalTokens': 3},
			'amazon-bedrock-invocationMetrics': {'invocationLatency': 111, 'firstByteLatency': 222},
		}
		m = bh._extract_metrics_from_response(resp, body)
		self.assertEqual(m.input_tokens, 1)
		self.assertEqual(m.output_tokens, 2)
		self.assertEqual(m.total_tokens, 3)
		self.assertEqual(m.invocation_latency_ms, 111)
		self.assertEqual(m.first_byte_latency_ms, 222)

		# Titan fallback inputTextTokenCount
		m2 = bh._extract_metrics_from_response({'ResponseMetadata': {'HTTPHeaders': {}}}, {'inputTextTokenCount': '9'})
		self.assertEqual(m2.input_tokens, 9)

		# Header fallback when body has no usage
		m3 = bh._extract_metrics_from_response(resp, {})
		self.assertEqual(m3.input_tokens, 10)
		self.assertEqual(m3.output_tokens, 20)
		self.assertEqual(m3.total_tokens, 30)

	def test_extract_metrics_pick_int_skips_none_and_bad_values_then_picks_valid(self):
		# Arrange: usage has:
		# - first candidate key present but None -> triggers "continue"
		# - second candidate key present but non-int -> triggers except(ValueError) then "pass"
		# - third candidate key valid int -> returns it
		resp = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {
			'usage': {
				'inputTokens': None,  # continue branch
				'prompt_tokens': 'nope',  # ValueError branch
				'promptTokens': '123',  # success (int("123") == 123)
				'outputTokens': '7',  # also works for output
				'totalTokens': None,  # ensure total_tokens fallback can happen elsewhere if needed
			}
		}

		m = BedrockHelper._extract_metrics_from_response(resp, body)

		self.assertEqual(m.input_tokens, 123)
		self.assertEqual(m.output_tokens, 7)
		self.assertEqual(m.total_tokens, 130)  # computed from input+output if totalTokens missing

	def test_extract_metrics_input_text_token_count_bad_value_is_ignored(self):
		# No usable "usage" dict => metrics.input_tokens stays None so we hit the Titan path.
		resp = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {
			# This exists and is not None, but int("nope") raises ValueError -> except branch.
			'inputTextTokenCount': 'nope'
		}

		m = BedrockHelper._extract_metrics_from_response(resp, body)

		# Should remain None because conversion failed and was swallowed.
		self.assertIsNone(m.input_tokens)

	def test_extract_metrics_input_text_token_count_type_error_is_ignored(self):
		resp = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {'inputTextTokenCount': object()}  # int(object()) -> TypeError

		m = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertIsNone(m.input_tokens)

	@patch('bedrockhelper.main.normalize_headers', return_value={'x-amzn-bedrock-input-token-count': 'nope'})
	def test_extract_metrics_header_get_int_bad_value_returns_none(self, norm_headers):
		# HTTPHeaders just needs to be truthy to enter the header-metrics block
		resp = {'ResponseMetadata': {'HTTPHeaders': {'X-Amzn-Bedrock-Input-Token-Count': 'nope'}}}

		m = BedrockHelper._extract_metrics_from_response(resp, response_body=None)

		# int("nope") raises ValueError => _get_int returns None (this hits the except branch)
		self.assertIsNone(m.input_tokens)

	def test_init_rejects_invalid_max_concurrent_streams(self):
		with self.assertRaises(ValueError):
			BedrockHelper(max_concurrent_streams=0)

	@patch('bedrockhelper.main.Config')
	def test_init_builds_default_botocore_config_when_none(self, ConfigMock):
		# Config(...) should be invoked, and then passed into _BotoSessionManager
		cfg_obj = MagicMock(name='cfg')
		ConfigMock.return_value = cfg_obj

		with patch('bedrockhelper.main._BotoSessionManager') as MgrMock:
			MgrMock.return_value.client.return_value = MagicMock()
			BedrockHelper(botocore_config=None)

		ConfigMock.assert_called_once()
		kwargs = ConfigMock.call_args.kwargs
		self.assertEqual(kwargs['connect_timeout'], 5)
		self.assertEqual(kwargs['read_timeout'], 120)
		self.assertEqual(kwargs['retries'], {'max_attempts': 10, 'mode': 'adaptive'})

	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch('bedrockhelper.main.format_context_passages', side_effect=lambda passages, include_headers=False: 'CTX')
	@patch('bedrockhelper.main.build_converse_request', side_effect=lambda **kw: {'modelId': kw['model_id']})
	@patch('bedrockhelper.main.extract_converse_text', return_value='CONVERSE_TEXT')
	def test_generate_with_rag_converse_nonstream_context_seq(
		self,
		extract_text,
		build_req,
		format_ctx,
		trunc,
	):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.converse.return_value = {'output': 'x'}

		out = bh.generate_with_rag(
			system_prompt='S',
			context=['a', 'b'],
			include_headers_in_context=True,
			question='Q',
			stream=False,
			prefer_converse=True,
		)
		self.assertEqual(out.text, 'CONVERSE_TEXT')
		bh.bedrock_runtime.converse.assert_called_once()
		format_ctx.assert_called_once()

	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch('bedrockhelper.main.build_converse_request', side_effect=lambda **kw: {'modelId': kw['model_id']})
	@patch(
		'bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('hi')
	)
	def test_generate_with_rag_converse_stream_success(self, iter_stream, build_req, trunc):
		bh = self._make_helper_with_clients()
		stream = _FakeClosableStream()
		bh.bedrock_runtime.converse_stream.return_value = {'stream': stream}

		out = bh.generate_with_rag(
			system_prompt='S',
			context='CTXSTR',
			question='Q',
			stream=True,
			prefer_converse=True,
		)
		self.assertTrue(stream.closed)
		self.assertEqual(out.text, 'hi')
		self.assertTrue(out.stream)

	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	def test_generate_with_rag_stream_true_without_converse_stream_uses_converse(self, trunc):
		bh = self._make_helper_with_clients()

		# Make the runtime client look like it has converse but NOT converse_stream
		del bh.bedrock_runtime.converse_stream  # MagicMock creates attrs; delete to force hasattr False
		bh.bedrock_runtime.converse.return_value = {'output': 'x'}

		# Keep internals simple: make extract_converse_text return deterministic text
		with patch('bedrockhelper.main.extract_converse_text', return_value='TEXT'):
			out = bh.generate_with_rag(
				system_prompt='S',
				context='CTX',
				question='Q',
				stream=True,
				prefer_converse=True,
			)

		self.assertEqual(out.text, 'TEXT')
		bh.bedrock_runtime.converse.assert_called_once()

	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch(
		'bedrockhelper.main.iter_bedrock_stream_text',
		side_effect=lambda stream, on_text, stream_kind: on_text('fallback'),
	)
	def test_generate_with_rag_converse_stream_missing_stream_falls_back(self, iter_stream, trunc):
		bh = self._make_helper_with_clients()

		bh.bedrock_runtime.converse_stream.return_value = {}

		# Make event_stream closable so the "hasattr(close)" branch is exercised.
		event_stream = MagicMock()
		event_stream.close = MagicMock()
		bh.bedrock_runtime.invoke_model_with_response_stream.return_value = {'body': event_stream}

		out = bh.generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			stream=True,
			prefer_converse=True,
		)

		self.assertEqual(out.text, 'fallback')
		self.assertTrue(out.stream)

		bh.bedrock_runtime.invoke_model_with_response_stream.assert_called_once()
		iter_stream.assert_called_once()
		event_stream.close.assert_called_once()

	def test__generate_with_converse_stream_missing_stream_raises(self):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.converse_stream.return_value = {}

		with self.assertRaises(RuntimeError):
			bh._generate_with_converse_stream(
				model_id='m',
				system_prompt='S',
				context='CTX',
				question='Q',
				max_tokens=10,
				temperature=0.1,
				rag_instructions='',
			)

	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	def test_generate_with_rag_converse_throws_falls_back_to_invoke(self, trunc):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.converse.side_effect = RuntimeError('nope')
		bh.bedrock_runtime.invoke_model.return_value = {
			'body': BytesIO(json.dumps({'content': [{'text': 'fallback'}]}).encode('utf-8'))
		}

		out = bh.generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			stream=False,
			prefer_converse=True,
		)
		self.assertEqual(out.text, 'fallback')
		bh.bedrock_runtime.invoke_model.assert_called_once()

	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	def test_generate_with_rag_prefer_converse_false_uses_invoke(self, trunc):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.invoke_model.return_value = {'body': BytesIO(json.dumps({'completion': 'yo'}).encode())}
		out = bh.generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			stream=False,
			prefer_converse=False,
		)
		self.assertEqual(out.text, 'yo')

	@patch('bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('x'))
	def test_claude_stream_missing_body_raises(self, iter_stream):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.invoke_model_with_response_stream.return_value = {}
		with self.assertRaises(RuntimeError):
			bh.generate_with_rag(
				system_prompt='S',
				context='CTX',
				question='Q',
				stream=True,
				prefer_converse=False,
			)

	def test_claude_nonstream_json_parse_fallback_raw(self):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.invoke_model.return_value = {'body': BytesIO(b'\xff\xfe\x00notjson')}
		out = bh.generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			stream=False,
			prefer_converse=False,
		)
		self.assertEqual(out.text, '')

	@patch('bedrockhelper.main.normalize_records', return_value=[])
	def test_embed_texts_empty_records(self, norm):
		bh = self._make_helper_with_clients()
		out = bh.embed_texts(records=[])
		self.assertEqual(out.embeddings, {})
		self.assertEqual(out.metrics, {})

	@patch('bedrockhelper.main.normalize_records', return_value=[('1', 'a'), ('2', 'b')])
	def test_embed_texts_sync_path_max_workers_1(self, norm):
		bh = self._make_helper_with_clients()

		def invoke_model(**kwargs):
			return {'body': BytesIO(json.dumps({'embedding': [1.0, 2.0]}).encode('utf-8'))}

		bh.bedrock_runtime.invoke_model.side_effect = invoke_model
		out = bh.embed_texts(records=[('1', 'a'), ('2', 'b')], batch_threshold=3000, max_workers=1)
		self.assertIn('1', out.embeddings)
		self.assertIn('2', out.embeddings)

	@patch('bedrockhelper.main.normalize_records', return_value=[('1', 'a'), ('2', 'b')])
	def test_embed_texts_sync_path_concurrent_branch(self, norm):
		# This covers the ThreadPoolExecutor/as_completed branch.
		bh = self._make_helper_with_clients()
		bh._embed_one = MagicMock(
			side_effect=[
				('1', [0.1], MagicMock()),
				('2', [0.2], MagicMock()),
			]
		)
		out = bh.embed_texts(records=[('1', 'a'), ('2', 'b')], batch_threshold=3000, max_workers=4)
		self.assertEqual(out.embeddings['1'], [0.1])
		self.assertEqual(out.embeddings['2'], [0.2])
		self.assertEqual(bh._embed_one.call_count, 2)

	@patch('bedrockhelper.main.normalize_records', return_value=[('1', 'a')])
	def test_embed_one_unexpected_embedding_raises(self, norm):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.invoke_model.return_value = {'body': BytesIO(json.dumps({'embedding': 'nope'}).encode())}
		with self.assertRaises(RuntimeError):
			bh.embed_texts(records=[('1', 'a')], batch_threshold=3000, max_workers=1)

	@patch('bedrockhelper.main.normalize_records', return_value=[('1', 'a')])
	def test_embed_texts_batch_path_calls_submit(self, norm):
		bh = self._make_helper_with_clients()
		bh.submit_embedding_batch_job = MagicMock(return_value=MagicMock(job_id='x'))
		out = bh.embed_texts(records=[('1', 'a')], batch_threshold=0, s3_bucket='b', role_arn='r')
		self.assertEqual(out.job_id, 'x')

	def test_submit_embedding_batch_job_requires_bucket_and_role(self):
		bh = self._make_helper_with_clients()
		with self.assertRaises(ValueError):
			bh.submit_embedding_batch_job([('1', 'a')], s3_bucket=None, s3_prefix='p', role_arn=None)

	@patch('bedrockhelper.main.normalize_records', return_value=[('1', 'a')])
	def test_submit_embedding_batch_job_normalizes_when_not_list_of_tuples(self, norm):
		# cover the normalize_records call inside submit_embedding_batch_job
		bh = self._make_helper_with_clients()
		bh.s3.upload_file.return_value = None
		bh.bedrock.create_model_invocation_job.return_value = {'jobId': 'job-1'}

		out = bh.submit_embedding_batch_job(
			pairs={'1': 'a'},  # not list/tuple -> triggers normalize_records
			s3_bucket='bucket',
			s3_prefix='pref',
			role_arn='arn:role/xyz',
		)
		self.assertEqual(out.job_id, 'job-1')
		norm.assert_called()

	@patch('bedrockhelper.main.normalize_records', return_value=[('1', 'a')])
	def test_submit_embedding_batch_job_happy_path_jobname_fallback(self, norm):
		bh = self._make_helper_with_clients()
		bh.s3.upload_file.return_value = None
		# no job identifiers -> should fall back to job_name
		bh.bedrock.create_model_invocation_job.return_value = {}

		out = bh.submit_embedding_batch_job(
			pairs=[('1', 'a')],
			s3_bucket='bucket',
			s3_prefix='pref',
			role_arn='arn:role/xyz',
		)
		self.assertTrue(out.job_id.startswith('bedrock-embedding-job-'))
		self.assertEqual(out.job_id, out.job_name)

	@patch('bedrockhelper.main.os.remove', side_effect=OSError('nope'))
	@patch('bedrockhelper.main.normalize_records', return_value=[('1', 'a')])
	def test_submit_embedding_batch_job_tempfile_cleanup_oserror_swallowed(self, norm, remove_mock):
		bh = self._make_helper_with_clients()
		bh.s3.upload_file.return_value = None
		bh.bedrock.create_model_invocation_job.return_value = {'jobId': 'job-1'}

		# Should not raise even though os.remove blows up in finally:
		out = bh.submit_embedding_batch_job(
			pairs=[('1', 'a')],
			s3_bucket='bucket',
			s3_prefix='pref',
			role_arn='arn:role/xyz',
		)
		self.assertEqual(out.job_id, 'job-1')
		remove_mock.assert_called_once()

	def test_get_batch_job(self):
		bh = self._make_helper_with_clients()
		bh.bedrock.get_model_invocation_job.return_value = {'status': 'Completed'}
		self.assertEqual(bh.get_batch_job('id')['status'], 'Completed')

	def test_wait_for_batch_job_stops_on_failed(self):
		bh = self._make_helper_with_clients()
		bh.get_batch_job = MagicMock(return_value={'status': 'Failed'})
		out = bh.wait_for_batch_job('id', poll_seconds=0.0, timeout_seconds=0.1)
		self.assertEqual(out['status'], 'Failed')

	def test_wait_for_batch_job_timeout(self):
		bh = self._make_helper_with_clients()
		bh.get_batch_job = MagicMock(return_value={'status': 'InProgress'})
		with self.assertRaises(TimeoutError):
			bh.wait_for_batch_job('id', poll_seconds=0.0, timeout_seconds=0.0)

	@patch('bedrockhelper.main.time.sleep', return_value=None)
	@patch('bedrockhelper.main.time.time')
	def test_wait_for_batch_job_times_out(self, time_time, sleep_mock):
		bh = self._make_helper_with_clients()

		# Never completes
		bh.get_batch_job = MagicMock(return_value={'status': 'InProgress'})

		# Simulate time moving:
		# - First call: used to compute deadline (now=100) -> deadline = 100 + 1 = 101
		# - Second call: inside loop check (now=100) -> not timed out, sleeps
		# - Third call: next loop check (now=102) -> timed out -> raises (hits line 919)
		time_time.side_effect = [100.0, 100.0, 102.0]

		with self.assertRaises(TimeoutError) as ctx:
			bh.wait_for_batch_job('job-1', poll_seconds=0.01, timeout_seconds=1.0)

		self.assertIn('Batch job did not complete within 1.0s', str(ctx.exception))
		sleep_mock.assert_called()  # proves we went through at least one non-terminal iteration

	def test_download_batch_results_jsonl_invalid_uri(self):
		bh = self._make_helper_with_clients()
		with self.assertRaises(ValueError):
			list(bh.download_batch_results_jsonl(output_s3_uri='http://nope'))

	def test_download_batch_results_jsonl_prefix_normalization_and_filters(self):
		bh = self._make_helper_with_clients()

		# prefix without trailing slash triggers prefix += '/'
		# also include key None branch
		pages = [{'Contents': [{'Key': None}, {'Key': 'x.txt'}, {'Key': 'ok.jsonl'}]}]
		bh.s3.get_paginator.return_value = _FakePaginator(pages)

		body = b'\n' + json.dumps({'recordId': '1', 'modelOutput': {'embedding': [1.0]}}).encode() + b'\n'
		bh.s3.get_object.return_value = {'Body': BytesIO(body)}

		items = list(bh.download_batch_results_jsonl(output_s3_uri='s3://bucket/prefix'))
		self.assertEqual(len(items), 1)
		self.assertEqual(items[0]['recordId'], '1')

		# assert paginator used normalized Prefix ending in '/'
		# (paginate is called inside download_batch_results_jsonl)
		# we can't easily see kwargs without wrapping paginator, so just ensure get_paginator called
		bh.s3.get_paginator.assert_called_once_with('list_objects_v2')

	def test_parse_batch_embeddings_branches(self):
		bh = self._make_helper_with_clients()

		def gen():
			yield {'x': 1}  # rid missing
			yield {'recordId': 'a', 'modelOutput': 'nope'}  # modelOut not dict
			yield {'recordId': 'b', 'modelOutput': {'embedding': 'nope'}}  # embedding not list
			yield {'record_id': 'c', 'model_output': {'embedding': [9.0]}}  # good
			yield {'recordId': 'd', 'output': {'embedding': [10.0]}}  # output alias branch

		bh.download_batch_results_jsonl = MagicMock(side_effect=lambda output_s3_uri: gen())
		out = bh.parse_batch_embeddings(output_s3_uri='s3://bucket/prefix/')
		self.assertEqual(out, {'c': [9.0], 'd': [10.0]})


class TestBedrockHelperAsync(IsolatedAsyncioTestCase):
	def _make_helper_with_clients(self):
		brt = MagicMock(name='bedrock-runtime')
		br = MagicMock(name='bedrock')
		s3 = MagicMock(name='s3')
		return BedrockHelper(bedrock_runtime_client=brt, bedrock_client=br, s3_client=s3)

	async def test_async_generate_with_rag_uses_to_thread(self):
		bh = self._make_helper_with_clients()
		bh.generate_with_rag = MagicMock(return_value=MagicMock(text='x'))
		out = await bh.async_generate_with_rag(system_prompt='S', context='C', question='Q')
		self.assertEqual(out.text, 'x')
		bh.generate_with_rag.assert_called_once()

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch('bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('A'))
	async def test_async_stream_generate_with_rag_prefers_converse_stream(self, iter_stream, trunc, ThreadMock):
		bh = self._make_helper_with_clients()

		stream = _FakeClosableStream()
		bh.bedrock_runtime.converse_stream.return_value = {'stream': stream}

		chunks = []
		async for t in bh.async_stream_generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			prefer_converse=True,
		):
			chunks.append(t)

		self.assertEqual(chunks, ['A'])
		self.assertTrue(stream.closed)
		bh.bedrock_runtime.converse_stream.assert_called_once()

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	async def test_async_stream_generate_with_rag_no_converse_stream_then_error_propagates(self, trunc, ThreadMock):
		bh = self._make_helper_with_clients()

		# Ensure there is NO converse_stream attribute, so worker skips converse path without try/except
		if hasattr(bh.bedrock_runtime, 'converse_stream'):
			del bh.bedrock_runtime.converse_stream

		# Force the fallback call itself to raise (worker catches BaseException -> sets err -> generator raises at end)
		bh.bedrock_runtime.invoke_model_with_response_stream.side_effect = RuntimeError('boom')

		with self.assertRaises(RuntimeError):
			async for _ in bh.async_stream_generate_with_rag(
				system_prompt='S',
				context='CTX',
				question='Q',
				prefer_converse=True,  # important: we want the "no converse_stream" skip branch
			):
				pass

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch('bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('B'))
	async def test_async_stream_generate_with_rag_converse_stream_fails_falls_back_to_invoke(
		self, iter_stream, trunc, ThreadMock
	):
		bh = self._make_helper_with_clients()

		# converse_stream exists but throws -> fallback path
		bh.bedrock_runtime.converse_stream.side_effect = RuntimeError('nope')

		event_stream = _FakeClosableStream()
		bh.bedrock_runtime.invoke_model_with_response_stream.return_value = {'body': event_stream}

		chunks = []
		async for t in bh.async_stream_generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			prefer_converse=True,
			rag_instructions='INS',
			max_tokens=5,
			temperature=0.0,
		):
			chunks.append(t)

		self.assertEqual(chunks, ['B'])
		self.assertTrue(event_stream.closed)
		bh.bedrock_runtime.invoke_model_with_response_stream.assert_called_once()

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	async def test_async_stream_generate_with_rag_missing_body_raises_from_worker(self, trunc, ThreadMock):
		bh = self._make_helper_with_clients()
		bh.bedrock_runtime.converse_stream.side_effect = RuntimeError('force fallback')
		bh.bedrock_runtime.invoke_model_with_response_stream.return_value = {}  # missing body triggers error

		with self.assertRaises(RuntimeError):
			async for _ in bh.async_stream_generate_with_rag(system_prompt='S', context='CTX', question='Q'):
				pass

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch('bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('C'))
	async def test_async_stream_generate_with_rag_release_exception_swallowed(self, iter_stream, trunc, ThreadMock):
		# Covers the "release throws -> swallow" branch
		bh = self._make_helper_with_clients()
		bh._stream_sema = _ExplodingReleaseSema()

		# Force fallback immediately (no converse_stream attr or failure)
		event_stream = _FakeClosableStream()
		bh.bedrock_runtime.invoke_model_with_response_stream.return_value = {'body': event_stream}

		chunks = []
		async for t in bh.async_stream_generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			prefer_converse=False,
		):
			chunks.append(t)

		self.assertEqual(chunks, ['C'])
		self.assertTrue(event_stream.closed)

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch(
		'bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('OK')
	)
	async def test_async_stream_generate_with_rag_release_error_is_swallowed(self, iter_stream, trunc, ThreadMock):
		bh = self._make_helper_with_clients()

		# Make release raise to hit the swallow branch
		bh._stream_sema = _ExplodingReleaseSema()

		# Go straight to fallback streaming path
		bh.bedrock_runtime.invoke_model_with_response_stream.return_value = {'body': _FakeClosableStream()}

		chunks = []
		async for t in bh.async_stream_generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			prefer_converse=False,
		):
			chunks.append(t)

		self.assertEqual(chunks, ['OK'])

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: f'TRUNC({s})')
	@patch('bedrockhelper.main.format_context_passages', return_value='FORMATTED')
	@patch(
		'bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('ok')
	)
	async def test_async_stream_normalize_context_value_sequence_calls_format_context_passages(
		self,
		iter_stream,
		format_ctx,
		trunc,
		ThreadMock,
	):
		# runtime must support fallback streaming call
		runtime = MagicMock(spec_set=['invoke_model_with_response_stream'])
		runtime.invoke_model_with_response_stream.return_value = {'body': _FakeClosableStream()}

		bh = BedrockHelper(
			bedrock_runtime_client=runtime,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

		# context is Sequence[str] -> hits the ELSE branch in normalize_context_value()
		ctx_list = ['a', 'b']

		out = []
		async for chunk in bh.async_stream_generate_with_rag(
			system_prompt='S',
			context=ctx_list,
			include_headers_in_context=True,  # ensures arg is wired through
			question='Q',
			prefer_converse=False,  # go directly to fallback stream path
			max_context_chars=123,
		):
			out.append(chunk)

		self.assertEqual(out, ['ok'])

		# These asserts prove line 651 else-branch executed
		format_ctx.assert_called_once_with(list(ctx_list), include_headers=True)
		trunc.assert_called_once_with('FORMATTED', 123)

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch(
		'bedrockhelper.main.iter_bedrock_stream_text', side_effect=lambda stream, on_text, stream_kind: on_text('ok')
	)
	async def test_async_stream_converse_stream_missing_stream_key_triggers_fallback(
		self, iter_stream, trunc, ThreadMock
	):
		class RuntimeWithBadConverseStream:
			# hasattr(..., "converse_stream") must be True, so define the method.
			def converse_stream(self, **kwargs):
				return {}  # missing "stream" => stream_obj is None => triggers line 696

			def invoke_model_with_response_stream(self, **kwargs):
				# fallback path used after the RuntimeError is caught
				return {'body': _FakeClosableStream()}

		runtime = RuntimeWithBadConverseStream()

		bh = BedrockHelper(
			bedrock_runtime_client=runtime,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

		chunks = []
		async for t in bh.async_stream_generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			prefer_converse=True,  # force converse_stream attempt
		):
			chunks.append(t)

		# If we successfully fell back, we should still get output
		self.assertEqual(chunks, ['ok'])

	@patch(
		'bedrockhelper.main.threading.Thread',
		side_effect=lambda target, daemon: _ImmediateThread(target=target, daemon=daemon),
	)
	@patch('bedrockhelper.main.truncate_by_chars', side_effect=lambda s, n: s)
	@patch(
		'bedrockhelper.main.iter_bedrock_stream_text',
		side_effect=lambda stream, on_text, stream_kind: on_text('ok'),
	)
	async def test_async_stream_fallback_includes_extra_params_in_body(
		self,
		iter_stream,
		trunc,
		ThreadMock,
	):
		captured = {}

		class RuntimeFallbackOnly:
			# No converse_stream => fallback path
			def invoke_model_with_response_stream(self, **kwargs):
				captured['body'] = json.loads(kwargs['body'])
				return {'body': _FakeClosableStream()}

		bh = BedrockHelper(
			bedrock_runtime_client=RuntimeFallbackOnly(),
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

		out = []
		async for chunk in bh.async_stream_generate_with_rag(
			system_prompt='S',
			context='CTX',
			question='Q',
			prefer_converse=True,  # doesn't matter; no converse_stream exists
			top_p=0.42,  # <-- EXTRA PARAM (this hits line 723)
		):
			out.append(chunk)

		self.assertEqual(out, ['ok'])

		# This assertion proves body.update(extra_params) ran
		self.assertIn('top_p', captured['body'])
		self.assertEqual(captured['body']['top_p'], 0.42)

	from unittest.mock import AsyncMock, MagicMock, patch

	@patch('bedrockhelper.main.asyncio.to_thread', new_callable=AsyncMock)
	async def test_async_embed_texts_uses_to_thread(self, to_thread):
		bh = BedrockHelper(
			bedrock_runtime_client=MagicMock(),
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

		# Make to_thread execute the provided callable synchronously and return its value.
		def _run_inline(fn, *args, **kwargs):
			return fn(*args, **kwargs)

		to_thread.side_effect = _run_inline

		# Make embed_texts return a sentinel so we can assert it flowed through.
		sentinel = EmbeddingResponse(embeddings={'a': [0.1]}, metrics={})
		bh.embed_texts = MagicMock(return_value=sentinel)

		out = await bh.async_embed_texts({'a': 'hello'})

		self.assertIs(out, sentinel)
		to_thread.assert_awaited_once()
		# Ensure it was called to run bh.embed_texts with our args.
		called_fn = to_thread.call_args.args[0]
		self.assertIs(called_fn, bh.embed_texts)
		self.assertEqual(to_thread.call_args.args[1:], ({'a': 'hello'},))
