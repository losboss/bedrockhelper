import json
from unittest import TestCase

from botocore.exceptions import ClientError

from stream_events import CONVERSE_EVENTS, INVOKE_EVENTS

from bedrockhelper.utils import (
	DEFAULT_TEMPERATURE,
	build_converse_request,
	extract_converse_text,
	format_context_passages,
	is_sampling_param_error,
	iter_stream_text,
	normalize_headers,
	normalize_records,
	normalize_s3_bucket_name,
	strip_sampling_params,
	supports_sampling_params,
	truncate_by_chars,
)


class TestNormalizeHeaders(TestCase):
	def test_string_keys_lowercased(self):
		result = normalize_headers({'Content-Type': 'text/html', 'Accept': 'json'})
		self.assertEqual(result, {'content-type': 'text/html', 'accept': 'json'})

	def test_numeric_keys_converted(self):
		result = normalize_headers({200: 'OK', 404: 'Not Found'})
		self.assertEqual(result, {'200': 'OK', '404': 'Not Found'})

	def test_mixed_keys(self):
		result = normalize_headers({'Header': 'val', 123: 'num'})
		self.assertEqual(result, {'header': 'val', '123': 'num'})

	def test_empty_dict(self):
		result = normalize_headers({})
		self.assertEqual(result, {})


class TestFormatContextPassages(TestCase):
	def test_default_formatting(self):
		result = format_context_passages(['passage one', 'passage two'])
		self.assertEqual(result, 'passage one\n\npassage two')

	def test_with_headers(self):
		result = format_context_passages(['p1', 'p2'], include_headers=True)
		self.assertIn('--- passage 1 ---', result)
		self.assertIn('--- passage 2 ---', result)

	def test_custom_header_template(self):
		result = format_context_passages(['p1'], include_headers=True, header_template='[{i}]')
		self.assertIn('[1]', result)

	def test_custom_separator(self):
		result = format_context_passages(['a', 'b'], separator=' | ')
		self.assertEqual(result, 'a | b')

	def test_empty_passages(self):
		result = format_context_passages([])
		self.assertEqual(result, '')

	def test_single_passage(self):
		result = format_context_passages(['only'])
		self.assertEqual(result, 'only')


class TestTruncateByChars(TestCase):
	def test_no_truncation(self):
		result = truncate_by_chars('hello world', 100)
		self.assertEqual(result, 'hello world')

	def test_truncates_at_limit(self):
		result = truncate_by_chars('hello world', 5)
		self.assertEqual(result, 'hello')

	def test_none_max_chars(self):
		result = truncate_by_chars('hello world', None)
		self.assertEqual(result, 'hello world')

	def test_zero_max_chars(self):
		result = truncate_by_chars('hello world', 0)
		self.assertEqual(result, '')

	def test_negative_max_chars(self):
		result = truncate_by_chars('hello world', -5)
		self.assertEqual(result, '')


class TestNormalizeRecords(TestCase):
	def test_dict_input(self):
		result = normalize_records({'k1': 'v1', 'k2': 'v2'})
		self.assertEqual(set(result), {('k1', 'v1'), ('k2', 'v2')})

	def test_tuple_sequence(self):
		result = normalize_records([('k1', 'v1'), ('k2', 'v2')])
		self.assertEqual(result, [('k1', 'v1'), ('k2', 'v2')])

	def test_mapping_sequence(self):
		result = normalize_records([{'k1': 'v1'}, {'k2': 'v2'}])
		self.assertEqual(result, [('k1', 'v1'), ('k2', 'v2')])

	def test_mapping_with_multiple_entries_raises(self):
		with self.assertRaises(ValueError) as ctx:
			normalize_records([{'k1': 'v1', 'k2': 'v2'}])
		self.assertIn('exactly 1 entry', str(ctx.exception))

	def test_unsupported_type_raises(self):
		with self.assertRaises(ValueError) as ctx:
			normalize_records(['string'])
		self.assertIn('Unsupported record item type', str(ctx.exception))

	def test_numeric_keys_converted(self):
		result = normalize_records({1: 'v1', 2: 'v2'})
		self.assertEqual(set(result), {('1', 'v1'), ('2', 'v2')})


class TestExtractConverseText(TestCase):
	def test_extracts_text(self):
		resp = {
			'output': {
				'message': {
					'content': [
						{'text': 'hello'},
						{'text': ' world'},
					]
				}
			}
		}
		result = extract_converse_text(resp)
		self.assertEqual(result, 'hello world')

	def test_empty_content(self):
		resp = {'output': {'message': {'content': []}}}
		result = extract_converse_text(resp)
		self.assertEqual(result, '')

	def test_missing_output(self):
		resp = {}
		result = extract_converse_text(resp)
		self.assertEqual(result, '')

	def test_non_dict_content_item(self):
		resp = {'output': {'message': {'content': ['not-a-dict', {'text': 'ok'}]}}}
		result = extract_converse_text(resp)
		self.assertEqual(result, 'ok')

	def test_non_string_text(self):
		resp = {'output': {'message': {'content': [{'text': 123}, {'text': 'ok'}]}}}
		result = extract_converse_text(resp)
		self.assertEqual(result, 'ok')

	def test_non_dict_output_returns_empty(self):
		resp = {'output': 'not-a-dict'}
		result = extract_converse_text(resp)
		self.assertEqual(result, '')


class TestBuildConverseRequest(TestCase):
	def test_basic_request(self):
		req = build_converse_request(
			model_id='test-model',
			system_prompt='You are helpful',
			context='Context here',
			question='Question?',
			max_tokens=100,
			temperature=0.7,
		)
		self.assertEqual(req['modelId'], 'test-model')
		self.assertEqual(req['system'][0]['text'], 'You are helpful')
		self.assertEqual(req['inferenceConfig']['maxTokens'], 100)
		self.assertEqual(req['inferenceConfig']['temperature'], 0.7)
		self.assertIn('Context here', req['messages'][0]['content'][0]['text'])
		self.assertIn('Question?', req['messages'][0]['content'][0]['text'])

	def test_empty_question(self):
		req = build_converse_request(
			model_id='test',
			system_prompt='sys',
			context='ctx',
			question='',
			max_tokens=50,
			temperature=0.5,
		)
		user_text = req['messages'][0]['content'][0]['text']
		self.assertIn('ctx', user_text)
		self.assertNotIn('Question:', user_text)

	def test_with_rag_instructions(self):
		req = build_converse_request(
			model_id='test',
			system_prompt='sys',
			context='ctx',
			question='q',
			max_tokens=50,
			temperature=0.5,
			rag_instructions='RAG instruction',
		)
		user_text = req['messages'][0]['content'][0]['text']
		self.assertIn('RAG instruction', user_text)

	def test_extra_params_merged(self):
		req = build_converse_request(
			model_id='test',
			system_prompt='sys',
			context='ctx',
			question='q',
			max_tokens=50,
			temperature=0.5,
			customField='value',
		)
		self.assertEqual(req['customField'], 'value')

	def test_library_default_applied_when_not_provided(self):
		req = build_converse_request(
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			system_prompt='sys',
			context='ctx',
			question='q',
			max_tokens=50,
		)
		self.assertEqual(req['inferenceConfig']['temperature'], DEFAULT_TEMPERATURE)
		self.assertEqual(req['inferenceConfig']['maxTokens'], 50)

	def test_explicit_none_omits_temperature(self):
		req = build_converse_request(
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			system_prompt='sys',
			context='ctx',
			question='q',
			max_tokens=50,
			temperature=None,
		)
		self.assertNotIn('temperature', req['inferenceConfig'])

	def test_temperature_dropped_for_incompatible_model(self):
		with self.assertLogs('bedrockhelper.utils', level='WARNING') as logs:
			req = build_converse_request(
				model_id='global.anthropic.claude-sonnet-5',
				system_prompt='sys',
				context='ctx',
				question='q',
				max_tokens=50,
				temperature=0.1,
			)
		self.assertNotIn('temperature', req['inferenceConfig'])
		self.assertIn('claude-sonnet-5', logs.output[0])

	def test_library_default_dropped_quietly_for_incompatible_model(self):
		# The caller never chose 0.1, so dropping it must not warn - otherwise
		# every default-path call to Sonnet 5 logs a spurious warning.
		with self.assertLogs('bedrockhelper.utils', level='DEBUG') as logs:
			req = build_converse_request(
				model_id='global.anthropic.claude-sonnet-5',
				system_prompt='sys',
				context='ctx',
				question='q',
				max_tokens=50,
			)
		self.assertNotIn('temperature', req['inferenceConfig'])
		self.assertEqual([r.levelname for r in logs.records], ['DEBUG'])

	def test_temperature_kept_for_compatible_model(self):
		req = build_converse_request(
			model_id='anthropic.claude-sonnet-4-5-20250929-v1:0',
			system_prompt='sys',
			context='ctx',
			question='q',
			max_tokens=50,
			temperature=0.3,
		)
		self.assertEqual(req['inferenceConfig']['temperature'], 0.3)


class TestSupportsSamplingParams(TestCase):
	def test_incompatible_families(self):
		for model_id in (
			'anthropic.claude-opus-4-7-20260210-v1:0',
			'us.anthropic.claude-opus-4-8-20260401-v1:0',
			'anthropic.claude-opus-5-20260601-v1:0',
			'global.anthropic.claude-sonnet-5-20260115-v1:0',
			'anthropic.claude-fable-5-20260601-v1:0',
			'anthropic.claude-mythos-5-20260601-v1:0',
		):
			with self.subTest(model_id=model_id):
				self.assertFalse(supports_sampling_params(model_id))

	def test_compatible_models(self):
		for model_id in (
			'anthropic.claude-sonnet-4-5-20250929-v1:0',
			'global.anthropic.claude-haiku-4-5-20251001-v1:0',
			'anthropic.claude-opus-4-6-20260101-v1:0',
			'amazon.titan-embed-text-v2:0',
			'amazon.nova-pro-v1:0',
			'test-model',
		):
			with self.subTest(model_id=model_id):
				self.assertTrue(supports_sampling_params(model_id))

	def test_case_insensitive(self):
		self.assertFalse(supports_sampling_params('ANTHROPIC.CLAUDE-SONNET-5-V1:0'))

	def test_empty_model_id(self):
		self.assertTrue(supports_sampling_params(''))


class TestStripSamplingParams(TestCase):
	def test_strips_from_inference_config(self):
		req = {
			'modelId': 'm',
			'inferenceConfig': {'maxTokens': 10, 'temperature': 0.5, 'topP': 0.9},
		}
		stripped, removed = strip_sampling_params(req)
		self.assertTrue(removed)
		self.assertEqual(stripped['inferenceConfig'], {'maxTokens': 10})

	def test_strips_top_level_keys(self):
		body = {'max_tokens': 10, 'temperature': 0.5, 'top_p': 0.9, 'system': 'sys'}
		stripped, removed = strip_sampling_params(body)
		self.assertTrue(removed)
		self.assertEqual(stripped, {'max_tokens': 10, 'system': 'sys'})

	def test_strips_from_additional_model_request_fields(self):
		req = {'additionalModelRequestFields': {'top_k': 5, 'other': 1}}
		stripped, removed = strip_sampling_params(req)
		self.assertTrue(removed)
		self.assertEqual(stripped['additionalModelRequestFields'], {'other': 1})

	def test_reports_no_change(self):
		req = {'modelId': 'm', 'inferenceConfig': {'maxTokens': 10}}
		stripped, removed = strip_sampling_params(req)
		self.assertFalse(removed)
		self.assertEqual(stripped, req)

	def test_does_not_mutate_input(self):
		req = {'inferenceConfig': {'maxTokens': 10, 'temperature': 0.5}}
		strip_sampling_params(req)
		self.assertEqual(req['inferenceConfig']['temperature'], 0.5)


class TestIsSamplingParamError(TestCase):
	@staticmethod
	def _validation_error(message: str) -> ClientError:
		return ClientError({'Error': {'Code': 'ValidationException', 'Message': message}}, 'Converse')

	def test_temperature_rejected(self):
		err = self._validation_error('This model does not support temperature.')
		self.assertTrue(is_sampling_param_error(err))

	def test_matches_live_bedrock_wording(self):
		# Verbatim message returned by Converse for claude-sonnet-5 on
		# 2026-07-29. Note the backticks around the parameter name.
		err = self._validation_error(
			'The model returned the following errors: `temperature` is deprecated for this model.'
		)
		self.assertTrue(is_sampling_param_error(err))

	def test_temperature_and_top_p_conflict(self):
		err = self._validation_error('temperature and top_p cannot both be specified for this model.')
		self.assertTrue(is_sampling_param_error(err))

	def test_unrelated_validation_error(self):
		err = self._validation_error('The provided model identifier is invalid.')
		self.assertFalse(is_sampling_param_error(err))

	def test_non_validation_client_error(self):
		err = ClientError({'Error': {'Code': 'ThrottlingException', 'Message': 'temperature'}}, 'Converse')
		self.assertFalse(is_sampling_param_error(err))

	def test_plain_exception(self):
		self.assertFalse(is_sampling_param_error(Exception('temperature is bad')))


class TestNormalizeS3BucketName(TestCase):
	def test_strips_s3_prefix(self):
		result = normalize_s3_bucket_name('s3://my-bucket')
		self.assertEqual(result, 'my-bucket')

	def test_no_prefix(self):
		result = normalize_s3_bucket_name('my-bucket')
		self.assertEqual(result, 'my-bucket')

	def test_strips_whitespace(self):
		result = normalize_s3_bucket_name('  my-bucket  ')
		self.assertEqual(result, 'my-bucket')

	def test_strips_prefix_and_whitespace(self):
		result = normalize_s3_bucket_name('  s3://my-bucket  ')
		self.assertEqual(result, 'my-bucket')


class TestIterStreamText(TestCase):
	def test_converse_yields_text_and_reports_metadata_after_message_stop(self):
		captured = []
		texts = list(iter_stream_text(iter(CONVERSE_EVENTS), on_metrics=captured.append))

		self.assertEqual(texts, ['Hello', ' world'])
		self.assertEqual(captured, [CONVERSE_EVENTS[-1]['metadata']])

	def test_invoke_yields_text_and_reports_invocation_metrics(self):
		captured = []
		texts = list(iter_stream_text(iter(INVOKE_EVENTS), on_metrics=captured.append))

		self.assertEqual(texts, ['Hello', ' world'])
		self.assertEqual(
			captured,
			[
				{
					'amazon-bedrock-invocationMetrics': {
						'inputTokenCount': 12,
						'outputTokenCount': 3,
						'invocationLatency': 321,
						'firstByteLatency': 45,
					}
				}
			],
		)

	def test_reads_events_after_message_stop(self):
		events = [*CONVERSE_EVENTS[:5], {'contentBlockDelta': {'delta': {'text': 'late'}}}, CONVERSE_EVENTS[5]]
		texts = list(iter_stream_text(iter(events), on_metrics=lambda m: None))
		self.assertEqual(texts, ['Hello', ' world', 'late'])

	def test_malformed_chunk_raises(self):
		with self.assertRaises(json.JSONDecodeError):
			list(iter_stream_text(iter([{'chunk': {'bytes': b'not json'}}]), on_metrics=lambda m: None))
