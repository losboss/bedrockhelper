import asyncio
import json
import os
import threading
from datetime import datetime, timedelta, UTC
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from botocore.config import Config
from botocore.exceptions import ClientError
from stream_events import CONVERSE_EVENTS, INVOKE_EVENTS

from bedrockhelper.main import BedrockHelper
from bedrockhelper.session import BotoSessionManager, is_expired_token, is_session_of_role
from bedrockhelper.models import (
	BatchJobResponse,
	EmbeddingResponse,
)


class TestIsExpiredToken(TestCase):
	def test_expired_token_error(self):
		err = ClientError({'Error': {'Code': 'ExpiredToken'}}, 'test')
		self.assertTrue(is_expired_token(err))

	def test_expired_token_exception(self):
		err = ClientError({'Error': {'Code': 'ExpiredTokenException'}}, 'test')
		self.assertTrue(is_expired_token(err))

	def test_invalid_client_token_id(self):
		err = ClientError({'Error': {'Code': 'InvalidClientTokenId'}}, 'test')
		self.assertTrue(is_expired_token(err))

	def test_unrecognized_client_exception(self):
		err = ClientError({'Error': {'Code': 'UnrecognizedClientException'}}, 'test')
		self.assertTrue(is_expired_token(err))

	def test_request_expired(self):
		err = ClientError({'Error': {'Code': 'RequestExpired'}}, 'test')
		self.assertTrue(is_expired_token(err))

	def test_other_client_error(self):
		err = ClientError({'Error': {'Code': 'AccessDenied'}}, 'test')
		self.assertFalse(is_expired_token(err))

	def test_non_client_error(self):
		err = ValueError('test')
		self.assertFalse(is_expired_token(err))


class TestBotoSessionManager(TestCase):
	@patch('bedrockhelper.session.boto3.Session')
	def test_no_role_arn(self, mock_session_cls):
		mock_session = Mock()
		mock_session_cls.return_value = mock_session

		mgr = BotoSessionManager(
			region_name='us-east-1',
			botocore_config=Mock(),
			role_arn=None,
		)

		sess = mgr._ensure_session()
		self.assertEqual(sess, mock_session)
		mock_session_cls.assert_called_once_with(region_name='us-east-1')

	@patch('bedrockhelper.session.boto3.Session')
	def test_invalidate(self, mock_session_cls):
		mgr = BotoSessionManager(
			region_name='us-east-1',
			botocore_config=Mock(),
		)
		mgr._boto3_session = Mock()
		mgr._clients = {'s3': Mock()}

		mgr.invalidate()

		self.assertIsNone(mgr._boto3_session)
		self.assertEqual(mgr._clients, {})

	@patch('bedrockhelper.session.boto3.Session')
	def test_client_caching(self, mock_session_cls):
		mock_session = Mock()
		mock_client = Mock()
		mock_session.client.return_value = mock_client
		mock_session_cls.return_value = mock_session

		mgr = BotoSessionManager(
			region_name='us-east-1',
			botocore_config=Mock(),
		)

		c1 = mgr.client('s3')
		c2 = mgr.client('s3')

		self.assertIs(c1, c2)
		mock_session.client.assert_called_once()

	@patch.object(BotoSessionManager, '_is_already_assumed_role', return_value=False)
	@patch('bedrockhelper.session.AssumeRoleCredentialFetcher')
	@patch.dict(
		os.environ,
		{
			'AWS_ACCESS_KEY_ID': 'SOURCE',
			'AWS_SECRET_ACCESS_KEY': 'SOURCE',
			'AWS_CONFIG_FILE': os.devnull,
			'AWS_SHARED_CREDENTIALS_FILE': os.devnull,
		},
		clear=True,
	)
	def test_assumed_role_credentials_refresh_after_expiry(self, mock_fetcher_cls, _mock_already):
		# Each AssumeRole returns new keys that are already inside botocore's
		# mandatory refresh window, so every read must go back to STS.
		fetches = []

		def fetch_credentials():
			fetches.append(None)
			expiry = datetime.now(UTC) + timedelta(seconds=30)
			return {
				'access_key': f'ASSUMED{len(fetches)}',
				'secret_key': 'secret',
				'token': f'token{len(fetches)}',
				'expiry_time': expiry.isoformat(),
			}

		mock_fetcher_cls.return_value.fetch_credentials = fetch_credentials

		mgr = BotoSessionManager(
			region_name='us-east-1',
			botocore_config=Config(),
			role_arn='arn:aws:iam::123456789012:role/test',
		)
		creds = mgr._ensure_session().get_credentials()

		first = creds.get_frozen_credentials().access_key
		second = creds.get_frozen_credentials().access_key

		self.assertTrue(first.startswith('ASSUMED'))
		self.assertNotEqual(first, second)


class TestIsSessionOfRole(TestCase):
	def test_matches_session_of_the_role(self):
		cases = [
			('arn:aws:sts::123456789012:assumed-role/Reader/s1', 'arn:aws:iam::123456789012:role/Reader'),
			('arn:aws:sts::123456789012:assumed-role/Reader/s1', 'arn:aws:iam::123456789012:role/team/a/Reader'),
			('arn:aws-cn:sts::123456789012:assumed-role/Reader/s1', 'arn:aws-cn:iam::123456789012:role/Reader'),
		]
		for caller, role in cases:
			with self.subTest(caller=caller, role=role):
				self.assertTrue(is_session_of_role(caller, role))

	def test_rejects_other_identities(self):
		role = 'arn:aws:iam::123456789012:role/Reader'
		cases = [
			'arn:aws:sts::999999999999:assumed-role/Reader/s1',  # other account
			'arn:aws:sts::123456789012:assumed-role/ReaderAdmin/s1',  # name shares a prefix
			'arn:aws:sts::123456789012:assumed-role/Writer/s1',
			'arn:aws-cn:sts::123456789012:assumed-role/Reader/s1',  # other partition
			'arn:aws:iam::123456789012:user/Reader',
			'arn:aws:sts::123456789012:federated-user/Reader',
			'',
			'not-an-arn',
		]
		for caller in cases:
			with self.subTest(caller=caller):
				self.assertFalse(is_session_of_role(caller, role))

	def test_rejects_a_target_that_is_not_a_role(self):
		caller = 'arn:aws:sts::123456789012:assumed-role/Reader/s1'
		self.assertFalse(is_session_of_role(caller, 'arn:aws:iam::123456789012:user/Reader'))
		self.assertFalse(is_session_of_role(caller, 'Reader'))


class TestAlreadyAssumedRole(TestCase):
	ROLE = 'arn:aws:iam::123456789012:role/Reader'

	def _manager(self):
		return BotoSessionManager(region_name='us-east-1', botocore_config=Config(), role_arn=self.ROLE)

	@patch('bedrockhelper.session.boto3.Session')
	def test_true_when_running_as_the_role(self, mock_session_cls):
		sts = mock_session_cls.return_value.client.return_value
		sts.get_caller_identity.return_value = {'Arn': 'arn:aws:sts::123456789012:assumed-role/Reader/pod'}
		self.assertTrue(self._manager()._is_already_assumed_role(self.ROLE))

	@patch('bedrockhelper.session.boto3.Session')
	def test_false_when_running_as_another_identity(self, mock_session_cls):
		sts = mock_session_cls.return_value.client.return_value
		sts.get_caller_identity.return_value = {'Arn': 'arn:aws:iam::123456789012:user/deploy'}
		self.assertFalse(self._manager()._is_already_assumed_role(self.ROLE))

	@patch('bedrockhelper.session.boto3.Session')
	def test_identity_lookup_failure_is_logged_and_assumes(self, mock_session_cls):
		sts = mock_session_cls.return_value.client.return_value
		sts.get_caller_identity.side_effect = ClientError(
			{'Error': {'Code': 'AccessDenied', 'Message': 'no'}}, 'GetCallerIdentity'
		)
		with self.assertLogs('bedrockhelper.session', level='WARNING'):
			self.assertFalse(self._manager()._is_already_assumed_role(self.ROLE))

	@patch.object(BotoSessionManager, '_is_already_assumed_role', return_value=True)
	@patch('bedrockhelper.session.AssumeRoleCredentialFetcher')
	@patch.dict(
		os.environ,
		{
			'AWS_ACCESS_KEY_ID': 'SOURCE',
			'AWS_SECRET_ACCESS_KEY': 'SOURCE',
			'AWS_CONFIG_FILE': os.devnull,
			'AWS_SHARED_CREDENTIALS_FILE': os.devnull,
		},
		clear=True,
	)
	def test_does_not_assume_the_role_it_already_has(self, mock_fetcher_cls, _mock_already):
		creds = self._manager()._ensure_session().get_credentials()

		mock_fetcher_cls.assert_not_called()
		self.assertEqual(creds.get_frozen_credentials().access_key, 'SOURCE')


class TestBedrockHelperInit(TestCase):
	def test_default_initialization(self):
		with patch.dict(os.environ, {}, clear=True):
			helper = BedrockHelper(
				bedrock_runtime_client=Mock(),
				bedrock_client=Mock(),
				s3_client=Mock(),
			)
			self.assertEqual(helper.rag_model_id, 'global.anthropic.claude-sonnet-5')

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

	def test_invalid_input_text_token_count_ignored(self):
		# Tests line 391-392: TypeError/ValueError in inputTextTokenCount
		response = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {'inputTextTokenCount': 'not-a-number'}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertIsNone(metrics.input_tokens)

	def test_invalid_invocation_metrics_values_ignored(self):
		# Tests lines 401-425: TypeError/ValueError in invocation metrics
		response = {'ResponseMetadata': {'HTTPHeaders': {}}}
		body = {
			'amazon-bedrock-invocationMetrics': {
				'inputTokenCount': 'bad',
				'outputTokenCount': [],
				'invocationLatency': {},
				'firstByteLatency': 'not-a-number',
			}
		}

		metrics = BedrockHelper._extract_metrics_from_response(response, body)

		self.assertIsNone(metrics.input_tokens)
		self.assertIsNone(metrics.output_tokens)
		self.assertIsNone(metrics.invocation_latency_ms)
		self.assertIsNone(metrics.first_byte_latency_ms)

	def test_invalid_header_values_ignored(self):
		# Tests lines 438-440: TypeError/ValueError in header parsing
		response = {
			'ResponseMetadata': {
				'HTTPHeaders': {
					'x-amzn-bedrock-input-token-count': 'not-a-number',
					'x-amzn-bedrock-output-token-count': [],
				}
			}
		}

		metrics = BedrockHelper._extract_metrics_from_response(response, None)

		self.assertIsNone(metrics.input_tokens)
		self.assertIsNone(metrics.output_tokens)

	def test_should_use_next_usage_key_when_first_is_not_a_number(self):
		body = {'usage': {'inputTokens': 'n/a', 'input_tokens': 7}}

		metrics = BedrockHelper._extract_metrics_from_response({}, body)

		self.assertEqual(metrics.input_tokens, 7)

	def test_should_keep_usage_counts_when_invocation_metrics_are_not_numbers(self):
		body = {
			'usage': {'inputTokens': 12, 'outputTokens': 3},
			'amazon-bedrock-invocationMetrics': {'inputTokenCount': 'bad', 'outputTokenCount': None},
		}

		metrics = BedrockHelper._extract_metrics_from_response({}, body)

		self.assertEqual((metrics.input_tokens, metrics.output_tokens), (12, 3))

	def test_should_leave_latency_unset_when_latency_ms_is_not_a_number(self):
		metrics = BedrockHelper._extract_metrics_from_response({}, {'metrics': {'latencyMs': 'soon'}})

		self.assertIsNone(metrics.invocation_latency_ms)


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

	def test_converse_errors_are_raised_not_retried_through_invoke_model(self):
		for code in ('AccessDeniedException', 'ThrottlingException', 'ValidationException'):
			with self.subTest(code=code):
				self.mock_runtime.reset_mock()
				error = ClientError({'Error': {'Code': code, 'Message': 'no'}}, 'Converse')
				self.mock_runtime.converse.side_effect = error

				with self.assertRaises(ClientError) as ctx:
					self.helper.generate_with_rag(system_prompt='prompt', context='context', question='question')

				self.assertIs(ctx.exception, error)
				self.mock_runtime.invoke_model.assert_not_called()

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

	def test_no_temperature_sent_by_default(self):
		# Default model is Sonnet 5, which has no temperature knob at all.
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		self.helper.generate_with_rag(system_prompt='prompt', context='context')

		inference_config = self.mock_runtime.converse.call_args[1]['inferenceConfig']
		self.assertNotIn('temperature', inference_config)

	def test_older_model_keeps_library_default_temperature(self):
		# Backward-compatibility guard: a caller on a model that still accepts
		# temperature, passing none, must keep getting 0.1 rather than silently
		# falling through to the model's own default of 1.0.
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='global.anthropic.claude-haiku-4-5-20251001-v1:0',
		)

		inference_config = self.mock_runtime.converse.call_args[1]['inferenceConfig']
		self.assertEqual(inference_config['temperature'], 0.1)

	def test_explicit_none_defers_to_model_default(self):
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			temperature=None,
		)

		inference_config = self.mock_runtime.converse.call_args[1]['inferenceConfig']
		self.assertNotIn('temperature', inference_config)

	def test_temperature_omitted_for_incompatible_model(self):
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='global.anthropic.claude-sonnet-5-20260115-v1:0',
			temperature=0.5,
		)

		inference_config = self.mock_runtime.converse.call_args[1]['inferenceConfig']
		self.assertNotIn('temperature', inference_config)

	def test_retries_without_sampling_params_on_validation_error(self):
		self.mock_runtime.converse.side_effect = [
			ClientError(
				{'Error': {'Code': 'ValidationException', 'Message': 'temperature is not supported'}},
				'Converse',
			),
			{
				'output': {'message': {'content': [{'text': 'answer'}]}},
				'ResponseMetadata': {'HTTPHeaders': {}},
			},
		]

		result = self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			temperature=0.5,
		)

		self.assertEqual(result.text, 'answer')
		self.assertEqual(self.mock_runtime.converse.call_count, 2)
		self.assertEqual(self.mock_runtime.converse.call_args_list[0][1]['inferenceConfig']['temperature'], 0.5)
		self.assertNotIn('temperature', self.mock_runtime.converse.call_args_list[1][1]['inferenceConfig'])
		self.mock_runtime.invoke_model.assert_not_called()

	def test_second_sampling_rejection_is_raised(self):
		error = ClientError(
			{'Error': {'Code': 'ValidationException', 'Message': 'temperature is not supported'}},
			'Converse',
		)
		self.mock_runtime.converse.side_effect = error

		with self.assertRaises(ClientError):
			self.helper.generate_with_rag(
				system_prompt='prompt',
				context='context',
				model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
				temperature=0.5,
			)

		self.assertEqual(self.mock_runtime.converse.call_count, 2)
		self.mock_runtime.invoke_model.assert_not_called()

	def test_converse_latency_is_reported(self):
		self.mock_runtime.converse.return_value = {
			'output': {'message': {'content': [{'text': 'answer'}]}},
			'usage': {'inputTokens': 12, 'outputTokens': 3, 'totalTokens': 15},
			'metrics': {'latencyMs': 321},
		}

		metrics = self.helper.generate_with_rag(system_prompt='p', context='c').metrics

		self.assertEqual((metrics.input_tokens, metrics.output_tokens, metrics.total_tokens), (12, 3, 15))
		self.assertEqual(metrics.invocation_latency_ms, 321)

	def test_invoke_model_omits_temperature_for_incompatible_model(self):
		self.mock_runtime.invoke_model.return_value = {
			'body': Mock(read=lambda: json.dumps({'content': [{'text': 'answer'}]}).encode()),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='anthropic.claude-opus-5-20260601-v1:0',
			temperature=0.5,
			prefer_converse=False,
		)

		body = json.loads(self.mock_runtime.invoke_model.call_args[1]['body'])
		self.assertNotIn('temperature', body)

	def test_invoke_model_retries_without_sampling_params(self):
		self.mock_runtime.invoke_model.side_effect = [
			ClientError(
				{'Error': {'Code': 'ValidationException', 'Message': 'temperature and top_p cannot both be specified'}},
				'InvokeModel',
			),
			{
				'body': Mock(read=lambda: json.dumps({'content': [{'text': 'answer'}]}).encode()),
				'ResponseMetadata': {'HTTPHeaders': {}},
			},
		]

		result = self.helper.generate_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			temperature=0.5,
			prefer_converse=False,
		)

		self.assertEqual(result.text, 'answer')
		self.assertEqual(self.mock_runtime.invoke_model.call_count, 2)
		retry_body = json.loads(self.mock_runtime.invoke_model.call_args_list[1][1]['body'])
		self.assertNotIn('temperature', retry_body)

	def test_unrelated_validation_error_is_not_retried(self):
		self.mock_runtime.converse.side_effect = ClientError(
			{'Error': {'Code': 'ValidationException', 'Message': 'invalid model identifier'}},
			'Converse',
		)

		with self.assertRaises(ClientError):
			self.helper.generate_with_rag(
				system_prompt='prompt',
				context='context',
				model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
				temperature=0.5,
			)

		self.assertEqual(self.mock_runtime.converse.call_count, 1)
		self.mock_runtime.invoke_model.assert_not_called()


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

	def test_converse_stream_error_is_raised_not_retried_through_invoke(self):
		error = ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'no'}}, 'ConverseStream')
		self.mock_runtime.converse_stream.side_effect = error

		with self.assertRaises(ClientError) as ctx:
			self.helper.stream_with_rag(system_prompt='prompt', context='context', question='question')

		self.assertIs(ctx.exception, error)
		self.mock_runtime.invoke_model_with_response_stream.assert_not_called()

	def test_converse_stream_result_has_token_metrics(self):
		self.mock_runtime.converse_stream.return_value = {'stream': iter(CONVERSE_EVENTS)}

		stream = self.helper.stream_with_rag(system_prompt='prompt', context='context')
		self.assertEqual(list(stream), ['Hello', ' world'])

		metrics = stream.result.metrics
		self.assertEqual(stream.result.text, 'Hello world')
		self.assertEqual((metrics.input_tokens, metrics.output_tokens, metrics.total_tokens), (12, 3, 15))
		self.assertEqual(metrics.invocation_latency_ms, 321)

	def test_invoke_stream_result_has_token_metrics(self):
		self.mock_runtime.invoke_model_with_response_stream.return_value = {'body': iter(INVOKE_EVENTS)}

		stream = self.helper.stream_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			prefer_converse=False,
		)
		self.assertEqual(list(stream), ['Hello', ' world'])

		metrics = stream.result.metrics
		self.assertEqual((metrics.input_tokens, metrics.output_tokens, metrics.total_tokens), (12, 3, 15))
		self.assertEqual((metrics.invocation_latency_ms, metrics.first_byte_latency_ms), (321, 45))

	def test_stream_is_closed_when_finished(self):
		event_stream = Mock()
		event_stream.__iter__ = Mock(return_value=iter(CONVERSE_EVENTS))
		self.mock_runtime.converse_stream.return_value = {'stream': event_stream}

		list(self.helper.stream_with_rag(system_prompt='prompt', context='context'))

		event_stream.close.assert_called_once()

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

	def test_converse_stream_retries_without_sampling_params(self):
		mock_stream = [
			{'contentBlockDelta': {'delta': {'text': 'hello'}}},
			{'messageStop': {'stopReason': 'done'}},
		]
		self.mock_runtime.converse_stream.side_effect = [
			ClientError(
				{'Error': {'Code': 'ValidationException', 'Message': 'temperature is not supported'}},
				'ConverseStream',
			),
			{
				'stream': iter(mock_stream),
				'ResponseMetadata': {'HTTPHeaders': {}},
			},
		]

		stream = self.helper.stream_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			temperature=0.5,
		)

		self.assertEqual(list(stream), ['hello'])
		self.assertEqual(self.mock_runtime.converse_stream.call_count, 2)
		self.assertNotIn(
			'temperature',
			self.mock_runtime.converse_stream.call_args_list[1][1]['inferenceConfig'],
		)
		self.mock_runtime.invoke_model_with_response_stream.assert_not_called()

	def test_converse_stream_omits_temperature_for_incompatible_model(self):
		self.mock_runtime.converse_stream.return_value = {
			'stream': iter([{'messageStop': {'stopReason': 'done'}}]),
			'ResponseMetadata': {'HTTPHeaders': {}},
		}

		self.helper.stream_with_rag(
			system_prompt='prompt',
			context='context',
			model_id='anthropic.claude-opus-5-20260601-v1:0',
			temperature=0.5,
		)

		inference_config = self.mock_runtime.converse_stream.call_args[1]['inferenceConfig']
		self.assertNotIn('temperature', inference_config)


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

	def test_should_raise_job_error_and_warn_when_temp_file_cannot_be_deleted(self):
		self.mock_bedrock.create_model_invocation_job.side_effect = RuntimeError('job rejected')

		with patch('bedrockhelper.main.os.remove', side_effect=PermissionError('locked')):
			with self.assertLogs('bedrockhelper.main', level='WARNING') as logs:
				with self.assertRaisesRegex(RuntimeError, 'job rejected'):
					self.helper.submit_embedding_batch_job(
						[('id1', 'text1')],
						s3_bucket='test-bucket',
						s3_prefix='prefix',
						role_arn='arn:aws:iam::123456789012:role/test',
					)

		self.assertIn('Could not delete batch input file', logs.output[0])

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


class TestAsyncStreamWithRAG(IsolatedAsyncioTestCase):
	def _helper(self, runtime, **kwargs):
		return BedrockHelper(bedrock_runtime_client=runtime, bedrock_client=Mock(), s3_client=Mock(), **kwargs)

	async def test_streams_text_then_resolves_final_metrics(self):
		runtime = Mock()
		runtime.converse_stream.return_value = {
			'stream': iter(CONVERSE_EVENTS),
			'ResponseMetadata': {'HTTPHeaders': {'x-amzn-requestid': 'req-1'}},
		}

		chunks, header_metrics, result_future = await self._helper(runtime).async_stream_with_rag(
			system_prompt='prompt', context='context'
		)
		texts = [t async for t in chunks]
		result = await asyncio.wait_for(result_future, timeout=5)

		self.assertEqual(texts, ['Hello', ' world'])
		self.assertEqual(header_metrics.header_metrics, {'x-amzn-requestid': 'req-1'})
		self.assertEqual(result.text, 'Hello world')
		self.assertEqual((result.metrics.input_tokens, result.metrics.output_tokens), (12, 3))

	async def test_error_before_streaming_is_raised_from_the_call(self):
		runtime = Mock()
		runtime.converse_stream.side_effect = ClientError(
			{'Error': {'Code': 'AccessDeniedException', 'Message': 'no'}}, 'ConverseStream'
		)

		with self.assertRaises(ClientError):
			await asyncio.wait_for(
				self._helper(runtime).async_stream_with_rag(system_prompt='prompt', context='context'), timeout=5
			)

	async def test_error_mid_stream_is_raised_from_the_iterator(self):
		def events():
			yield CONVERSE_EVENTS[1]
			raise RuntimeError('connection reset')

		runtime = Mock()
		runtime.converse_stream.return_value = {'stream': events()}

		chunks, _, result_future = await self._helper(runtime).async_stream_with_rag(
			system_prompt='prompt', context='context'
		)
		received = []
		with self.assertRaisesRegex(RuntimeError, 'connection reset'):
			async for t in chunks:
				received.append(t)

		self.assertEqual(received, ['Hello'])
		with self.assertRaisesRegex(RuntimeError, 'connection reset'):
			await asyncio.wait_for(result_future, timeout=5)

	async def test_failed_streams_release_their_concurrency_slot(self):
		runtime = Mock()
		runtime.converse_stream.side_effect = ClientError(
			{'Error': {'Code': 'ThrottlingException', 'Message': 'slow down'}}, 'ConverseStream'
		)
		helper = self._helper(runtime, max_concurrent_streams=1)

		for _ in range(3):
			with self.assertRaises(ClientError):
				await asyncio.wait_for(helper.async_stream_with_rag(system_prompt='p', context='c'), timeout=5)

	async def _worker_finished(self, helper):
		# With max_concurrent_streams=1, the slot frees only once the worker has
		# queued its last callback; two loop turns then run those callbacks.
		self.assertTrue(await asyncio.to_thread(helper._stream_sema.acquire, timeout=5))
		helper._stream_sema.release()
		await asyncio.sleep(0)
		await asyncio.sleep(0)

	def _record_loop_errors(self):
		errors = []
		asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: errors.append(ctx.get('exception')))
		return errors

	async def test_should_not_raise_on_loop_when_result_future_is_cancelled_mid_stream(self):
		errors = self._record_loop_errors()
		release = threading.Event()

		def events():
			yield CONVERSE_EVENTS[1]
			release.wait(5)
			yield from CONVERSE_EVENTS[2:]

		runtime = Mock()
		runtime.converse_stream.return_value = {'stream': events()}
		helper = self._helper(runtime, max_concurrent_streams=1)

		chunks, _, result_future = await helper.async_stream_with_rag(system_prompt='p', context='c')
		result_future.cancel()
		release.set()
		self.assertEqual([t async for t in chunks], ['Hello', ' world'])
		await self._worker_finished(helper)

		self.assertEqual(errors, [])

	async def test_should_not_raise_on_loop_when_the_call_is_cancelled_before_headers(self):
		errors = self._record_loop_errors()
		started = threading.Event()
		release = threading.Event()

		def converse_stream(**_):
			started.set()
			release.wait(5)
			return {'stream': iter(CONVERSE_EVENTS)}

		runtime = Mock()
		runtime.converse_stream.side_effect = converse_stream
		helper = self._helper(runtime, max_concurrent_streams=1)

		call = asyncio.create_task(helper.async_stream_with_rag(system_prompt='p', context='c'))
		self.assertTrue(await asyncio.to_thread(started.wait, 5))
		call.cancel()
		release.set()
		with self.assertRaises(asyncio.CancelledError):
			await call
		await self._worker_finished(helper)

		self.assertEqual(errors, [])
