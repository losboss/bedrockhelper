import json
import os
from unittest import TestCase
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from bedrockhelper.main import (
	BedrockHelper,
	_BotoSessionManager,
	_is_expired_token,
	_make_on_tail,
)
from bedrockhelper.models import (
	BatchJobResponse,
	EmbeddingResponse,
	RAGStream,
)


class TestIsExpiredToken(TestCase):
	def test_expired_token_error(self):
		err = ClientError({'Error': {'Code': 'ExpiredToken'}}, 'test')
		self.assertTrue(_is_expired_token(err))

	def test_expired_token_exception(self):
		err = ClientError({'Error': {'Code': 'ExpiredTokenException'}}, 'test')
		self.assertTrue(_is_expired_token(err))

	def test_invalid_client_token_id(self):
		err = ClientError({'Error': {'Code': 'InvalidClientTokenId'}}, 'test')
		self.assertTrue(_is_expired_token(err))

	def test_unrecognized_client_exception(self):
		err = ClientError({'Error': {'Code': 'UnrecognizedClientException'}}, 'test')
		self.assertTrue(_is_expired_token(err))

	def test_request_expired(self):
		err = ClientError({'Error': {'Code': 'RequestExpired'}}, 'test')
		self.assertTrue(_is_expired_token(err))

	def test_other_client_error(self):
		err = ClientError({'Error': {'Code': 'AccessDenied'}}, 'test')
		self.assertFalse(_is_expired_token(err))

	def test_non_client_error(self):
		err = ValueError('test')
		self.assertFalse(_is_expired_token(err))


class TestMakeOnTail(TestCase):
	def test_no_stream_in_ref(self):
		stream_ref = {}
		on_tail = _make_on_tail(stream_ref)
		on_tail({'usage': {'tokens': 10}})
		# Should not raise

	def test_sets_initial_tail_body(self):
		mock_stream = Mock(spec=RAGStream)
		stream_ref = {'stream': mock_stream}
		on_tail = _make_on_tail(stream_ref)

		msg = {'usage': {'tokens': 10}}
		on_tail(msg)
		mock_stream._set_tail_body.assert_called_once_with(msg)

	def test_keeps_usage_if_present(self):
		mock_stream = Mock(spec=RAGStream)
		mock_stream._tail_body = {'usage': {'tokens': 10}}
		stream_ref = {'stream': mock_stream}
		on_tail = _make_on_tail(stream_ref)

		on_tail({'other': 'data'})
		# Should not replace existing usage
		self.assertEqual(mock_stream._set_tail_body.call_count, 0)

	def test_upgrades_to_usage(self):
		mock_stream = Mock(spec=RAGStream)
		mock_stream._tail_body = {'other': 'data'}
		stream_ref = {'stream': mock_stream}
		on_tail = _make_on_tail(stream_ref)

		new_msg = {'usage': {'tokens': 10}}
		on_tail(new_msg)
		mock_stream._set_tail_body.assert_called_once_with(new_msg)

	def test_updates_latest_without_usage(self):
		mock_stream = Mock(spec=RAGStream)
		mock_stream._tail_body = {'other': 'old'}
		stream_ref = {'stream': mock_stream}
		on_tail = _make_on_tail(stream_ref)

		new_msg = {'other': 'new'}
		on_tail(new_msg)
		mock_stream._set_tail_body.assert_called_once_with(new_msg)


class TestBotoSessionManager(TestCase):
	@patch('bedrockhelper.main.boto3.Session')
	def test_no_role_arn(self, mock_session_cls):
		mock_session = Mock()
		mock_session_cls.return_value = mock_session

		mgr = _BotoSessionManager(
			region_name='us-east-1',
			botocore_config=Mock(),
			role_arn=None,
		)

		sess = mgr._ensure_session()
		self.assertEqual(sess, mock_session)
		mock_session_cls.assert_called_once_with(region_name='us-east-1')

	@patch('bedrockhelper.main.boto3.Session')
	def test_invalidate(self, mock_session_cls):
		mgr = _BotoSessionManager(
			region_name='us-east-1',
			botocore_config=Mock(),
		)
		mgr._boto3_session = Mock()
		mgr._clients = {'s3': Mock()}

		mgr.invalidate()

		self.assertIsNone(mgr._boto3_session)
		self.assertEqual(mgr._clients, {})

	@patch('bedrockhelper.main.boto3.Session')
	def test_client_caching(self, mock_session_cls):
		mock_session = Mock()
		mock_client = Mock()
		mock_session.client.return_value = mock_client
		mock_session_cls.return_value = mock_session

		mgr = _BotoSessionManager(
			region_name='us-east-1',
			botocore_config=Mock(),
		)

		c1 = mgr.client('s3')
		c2 = mgr.client('s3')

		self.assertIs(c1, c2)
		mock_session.client.assert_called_once()


class TestBedrockHelperInit(TestCase):
	def test_default_initialization(self):
		with patch.dict(os.environ, {}, clear=True):
			helper = BedrockHelper(
				bedrock_runtime_client=Mock(),
				bedrock_client=Mock(),
				s3_client=Mock(),
			)
			self.assertEqual(helper.rag_model_id, 'global.anthropic.claude-sonnet-4-5-20250929-v1:0')

	def test_env_var_model_ids(self):
		with patch.dict(
			os.environ,
			{
				'AWS_MODEL_ID': 'custom-rag',
				'AWS_EMBEDDING_MODEL_ID': 'custom-embed',
			},
		):
			helper = BedrockHelper(
				bedrock_runtime_client=Mock(),
				bedrock_client=Mock(),
				s3_client=Mock(),
			)
			self.assertEqual(helper.rag_model_id, 'custom-rag')
			self.assertEqual(helper.embedding_model_id, 'custom-embed')

	def test_invalid_max_concurrent_streams(self):
		with self.assertRaises(ValueError):
			BedrockHelper(
				max_concurrent_streams=0,
				bedrock_runtime_client=Mock(),
				bedrock_client=Mock(),
				s3_client=Mock(),
			)


class TestExtractMetricsFromResponse(TestCase):
	def test_extracts_usage_from_body(self):
		response = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {'usage': {'inputTokens': 10, 'outputTokens': 20}}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertEqual(metrics.input_tokens, 10)
		self.assertEqual(metrics.output_tokens, 20)
		self.assertEqual(metrics.total_tokens, 30)

	def test_extracts_usage_snake_case(self):
		response = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {'usage': {'input_tokens': 15, 'output_tokens': 25}}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertEqual(metrics.input_tokens, 15)
		self.assertEqual(metrics.output_tokens, 25)

	def test_extracts_titan_input_text_token_count(self):
		response = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {'inputTextTokenCount': 50}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertEqual(metrics.input_tokens, 50)

	def test_extracts_invocation_metrics_from_body(self):
		response = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {
			'amazon-bedrock-invocationMetrics': {
				'invocationLatency': 100,
				'firstByteLatency': 50,
			}
		}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertEqual(metrics.invocation_latency_ms, 100)
		self.assertEqual(metrics.first_byte_latency_ms, 50)

	def test_extracts_header_metrics(self):
		response = {
			'ResponseMetadata': {
				'HTTPHeaders': {
					'x-amzn-bedrock-input-token-count': '30',
					'x-amzn-bedrock-output-token-count': '40',
					'x-amzn-bedrock-invocation-latency': '200',
					'x-amzn-bedrock-first-byte-latency': '75',
				}
			}
		}

		metrics = BedrockHelper._extract_metrics_from_response(response, None)

		self.assertEqual(metrics.input_tokens, 30)
		self.assertEqual(metrics.output_tokens, 40)
		self.assertEqual(metrics.invocation_latency_ms, 200)
		self.assertEqual(metrics.first_byte_latency_ms, 75)

	def test_body_metrics_preferred_over_headers(self):
		response = {'ResponseMetadata': {'HTTPHeaders': {'x-amzn-bedrock-input-token-count': '100'}}}
		body = {'usage': {'inputTokens': 50}}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertEqual(metrics.input_tokens, 50)

	def test_invalid_token_count_ignored(self):
		response = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {'usage': {'inputTokens': 'invalid'}}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertIsNone(metrics.input_tokens)

	def test_empty_response(self):
		metrics = BedrockHelper._extract_metrics_from_response({}, None)

		self.assertIsNone(metrics.input_tokens)
		self.assertIsNone(metrics.output_tokens)


class TestGenerateWithRAG(TestCase):
	def setUp(self):
		self.mock_runtime = Mock()
		self.mock_bedrock = Mock()
		self.mock_s3 = Mock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=self.mock_runtime,
			bedrock_client=self.mock_bedrock,
			s3_client=self.mock_s3,
		)

	def test_converse_success(self):
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		result = self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			question='question',
		)

		self.assertEqual(result.text, 'answer')
		self.assertFalse(result.stream)

	def test_converse_fallback_on_error(self):
		self.mock_runtime.converse.side_effect = Exception('converse failed')
		self.mock_runtime.invoke_model.return_value = {
			'body': Mock(read=lambda: json.dumps({'content': [{'text': 'fallback answer'}]}).encode()),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		result = self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			question='question',
		)

		self.assertEqual(result.text, 'fallback answer')

	def test_context_as_list(self):
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		result = self.helper.generate_with_rag(
			system_prompt='prompt',
			context=['passage1', 'passage2'],
			question='question',
		)

		self.assertEqual(result.text, 'answer')

	def test_custom_model_id(self):
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			question='question',
			model_id='custom-model',
		)

		call_kwargs = self.mock_runtime.converse.call_args[1]
		self.assertEqual(call_kwargs['modelId'], 'custom-model')

	def test_prefer_converse_false(self):
		self.mock_runtime.invoke_model.return_value = {
			'body': Mock(read=lambda: json.dumps({'content': [{'text': 'invoke answer'}]}).encode()),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		result = self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			question='question',
			prefer_converse=False,
		)

		self.assertEqual(result.text, 'invoke answer')
		self.mock_runtime.converse.assert_not_called()


class TestExtractTextFromClaude(TestCase):
	def test_content_list(self):
		body = {'content': [{'text': 'hello'}, {'text': ' world'}]}
		text = BedrockHelper._extract_text_from_claude(body)
		self.assertEqual(text, 'hello world')

	def test_completion_field(self):
		body = {'completion': 'direct completion'}
		text = BedrockHelper._extract_text_from_claude(body)
		self.assertEqual(text, 'direct completion')

	def test_string_body(self):
		text = BedrockHelper._extract_text_from_claude('plain string')
		self.assertEqual(text, 'plain string')

	def test_non_text_items_skipped(self):
		body = {'content': [{'image': 'data'}, {'text': 'ok'}]}
		text = BedrockHelper._extract_text_from_claude(body)
		self.assertEqual(text, 'ok')

	def test_empty_body(self):
		text = BedrockHelper._extract_text_from_claude({})
		self.assertEqual(text, '')


class TestStreamWithRAG(TestCase):
	def setUp(self):
		self.mock_runtime = Mock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=self.mock_runtime,
			bedrock_client=Mock(),
			s3_client=Mock(),
		)

	def test_converse_stream_success(self):
		mock_stream = [
			{'contentBlockDelta': {'delta': {'text': 'hello'}}},
			{'messageStop': {'stopReason': 'done'}},
		]

		self.mock_runtime.converse_stream.return_value = {
			'stream': iter(mock_stream),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		stream = self.helper.stream_with_rag(
			system_prompt='prompt',
			context='context',
			question='question',
		)

		texts = list(stream)
		self.assertEqual(texts, ['hello'])
		self.assertEqual(stream.result.text, 'hello')

	def test_fallback_to_invoke_stream(self):
		self.mock_runtime.converse_stream.side_effect = Exception('failed')

		mock_events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'fallback'}}).encode()}},
		]
		self.mock_runtime.invoke_model_with_response_stream.return_value = {
			'body': iter(mock_events),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		stream = self.helper.stream_with_rag(
			system_prompt='prompt',
			context='context',
			question='question',
		)

		texts = list(stream)
		self.assertEqual(texts, ['fallback'])

	def test_prefer_converse_false(self):
		mock_events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'invoke'}}).encode()}},
		]
		self.mock_runtime.invoke_model_with_response_stream.return_value = {
			'body': iter(mock_events),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		stream = self.helper.stream_with_rag(
			system_prompt='prompt',
			context='context',
			question='question',
			prefer_converse=False,
		)

		texts = list(stream)
		self.assertEqual(texts, ['invoke'])
		self.mock_runtime.converse_stream.assert_not_called()


class TestEmbedTexts(TestCase):
	def setUp(self):
		self.mock_runtime = Mock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=self.mock_runtime,
			bedrock_client=Mock(),
			s3_client=Mock(),
		)

	def test_small_batch_sync(self):
		self.mock_runtime.invoke_model.return_value = {
			'body': Mock(read=lambda: json.dumps({'embedding': [0.1, 0.2]}).encode()),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		result = self.helper.embed_texts({'id1': 'text1'})

		self.assertIsInstance(result, EmbeddingResponse)
		self.assertIn('id1', result.embeddings)
		self.assertEqual(result.embeddings['id1'], [0.1, 0.2])

	def test_large_batch_triggers_batch_job(self):
		large_records = {f'id{i}': f'text{i}' for i in range(4000)}

		with self.assertRaises(ValueError) as ctx:
			self.helper.embed_texts(
				large_records,
				batch_threshold=3000,
			)

		self.assertIn('s3_bucket', str(ctx.exception))

	def test_empty_records(self):
		result = self.helper.embed_texts({})

		self.assertIsInstance(result, EmbeddingResponse)
		self.assertEqual(result.embeddings, {})

	def test_embed_one_invalid_response(self):
		self.mock_runtime.invoke_model.return_value = {
			'body': Mock(read=lambda: json.dumps({'invalid': 'data'}).encode()),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		with self.assertRaises(RuntimeError):
			self.helper._embed_one('id1', 'text1')


class TestBatchEmbeddings(TestCase):
	def setUp(self):
		self.mock_s3 = Mock()
		self.mock_bedrock = Mock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=Mock(),
			bedrock_client=self.mock_bedrock,
			s3_client=self.mock_s3,
		)

	def test_submit_batch_job(self):
		self.mock_bedrock.create_model_invocation_job.return_value = {
			'jobArn': 'arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job'
		}

		result = self.helper.submit_embedding_batch_job(
			[('id1', 'text1'), ('id2', 'text2')],
			s3_bucket='test-bucket',
			s3_prefix='prefix',
			role_arn='arn:aws:iam::123456789012:role/test',
		)

		self.assertIsInstance(result, BatchJobResponse)
		self.assertIn('test-job', result.job_id)
		self.mock_s3.upload_file.assert_called_once()

	def test_submit_batch_job_no_bucket(self):
		with self.assertRaises(ValueError) as ctx:
			self.helper.submit_embedding_batch_job(
				[('id1', 'text1')],
				s3_bucket=None,
				s3_prefix='prefix',
				role_arn='arn',
			)

		self.assertIn('s3_bucket', str(ctx.exception))

	def test_submit_batch_job_no_role_arn(self):
		with self.assertRaises(ValueError) as ctx:
			self.helper.submit_embedding_batch_job(
				[('id1', 'text1')],
				s3_bucket='bucket',
				s3_prefix='prefix',
				role_arn=None,
			)

		self.assertIn('role_arn', str(ctx.exception))

	def test_get_batch_job(self):
		self.mock_bedrock.get_model_invocation_job.return_value = {'status': 'InProgress'}

		result = self.helper.get_batch_job('job-123')

		self.assertEqual(result['status'], 'InProgress')
		self.mock_bedrock.get_model_invocation_job.assert_called_once_with(jobIdentifier='job-123')

	def test_wait_for_batch_job_completed(self):
		self.mock_bedrock.get_model_invocation_job.return_value = {'status': 'Completed'}

		result = self.helper.wait_for_batch_job('job-123', poll_seconds=0.01)

		self.assertEqual(result['status'], 'Completed')

	def test_wait_for_batch_job_timeout(self):
		self.mock_bedrock.get_model_invocation_job.return_value = {'status': 'InProgress'}

		with self.assertRaises(TimeoutError):
			self.helper.wait_for_batch_job('job-123', poll_seconds=0.01, timeout_seconds=0.05)


class TestDownloadBatchResults(TestCase):
	def setUp(self):
		self.mock_s3 = Mock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=Mock(),
			bedrock_client=Mock(),
			s3_client=self.mock_s3,
		)

	def test_download_batch_results_jsonl(self):
		mock_paginator = Mock()
		mock_paginator.paginate.return_value = [
			{
				'Contents': [
					{'Key': 'prefix/output.jsonl'},
					{'Key': 'prefix/other.txt'},
				]
			}
		]
		self.mock_s3.get_paginator.return_value = mock_paginator

		jsonl_data = json.dumps({'recordId': 'id1', 'modelOutput': {'embedding': [0.1]}})
		self.mock_s3.get_object.return_value = {'Body': Mock(read=lambda: jsonl_data.encode())}

		results = list(self.helper.download_batch_results_jsonl(output_s3_uri='s3://bucket/prefix/'))

		self.assertEqual(len(results), 1)
		self.assertEqual(results[0]['recordId'], 'id1')

	def test_download_batch_results_invalid_uri(self):
		with self.assertRaises(ValueError):
			list(self.helper.download_batch_results_jsonl(output_s3_uri='invalid-uri'))

	def test_parse_batch_embeddings(self):
		with patch.object(self.helper, 'download_batch_results_jsonl') as mock_download:
			mock_download.return_value = [
				{'recordId': 'id1', 'modelOutput': {'embedding': [0.1, 0.2]}},
				{'recordId': 'id2', 'modelOutput': {'embedding': [0.3, 0.4]}},
				{'recordId': 'id3'},  # No embedding
			]

			result = self.helper.parse_batch_embeddings(output_s3_uri='s3://bucket/prefix/')

			self.assertEqual(len(result), 2)
			self.assertEqual(result['id1'], [0.1, 0.2])
			self.assertEqual(result['id2'], [0.3, 0.4])


class TestCallWithRefresh(TestCase):
	def setUp(self):
		self.helper = BedrockHelper(
			bedrock_runtime_client=Mock(),
			bedrock_client=Mock(),
			s3_client=Mock(),
		)

	def test_successful_call(self):
		mock_fn = Mock(return_value='success')

		result = self.helper._call_with_refresh(mock_fn, 'arg1', key='value')

		self.assertEqual(result, 'success')
		mock_fn.assert_called_once_with('arg1', key='value')

	def test_non_expired_error_raised(self):
		mock_fn = Mock(side_effect=ValueError('error'))

		with self.assertRaises(ValueError):
			self.helper._call_with_refresh(mock_fn)

	def test_expired_token_refreshes_and_retries(self):
		mock_fn = Mock(side_effect=[ClientError({'Error': {'Code': 'ExpiredToken'}}, 'test'), 'success'])

		with patch.object(self.helper._session_mgr, 'invalidate') as mock_invalidate:
			result = self.helper._call_with_refresh(mock_fn)

		self.assertEqual(result, 'success')
		mock_invalidate.assert_called_once()
		self.assertEqual(mock_fn.call_count, 2)
