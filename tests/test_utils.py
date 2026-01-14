import json
from unittest import TestCase

from bedrockhelper.utils import (
	_iter_stream_with_callback,
	build_converse_request,
	extract_converse_text,
	format_context_passages,
	iter_bedrock_stream_text,
	iter_bedrock_stream_text_gen_with_tail,
	normalize_headers,
	normalize_records,
	normalize_s3_bucket_name,
	truncate_by_chars,
)


class TestIterBedrockStreamText(TestCase):
	def test_invoke_stream_yields_text(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'hello'}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': ' world'}, 'type': 'end'}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke')
		self.assertEqual(texts, ['hello', ' world'])

	def test_invoke_stream_stops_on_stop_type(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'a'}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'type': 'message_stop'}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ignored'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke')
		self.assertEqual(texts, ['a'])

	def test_invoke_stream_custom_stop_types(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'a'}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'type': 'custom_stop'}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ignored'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke', invoke_stop_types=('custom_stop',))
		self.assertEqual(texts, ['a'])

	def test_invoke_stream_skips_empty_chunk(self):
		events = [
			{'chunk': {}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke')
		self.assertEqual(texts, ['ok'])

	def test_invoke_stream_skips_invalid_json(self):
		events = [
			{'chunk': {'bytes': b'invalid-json'}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke')
		self.assertEqual(texts, ['ok'])

	def test_invoke_stream_skips_non_string_text(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 123}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke')
		self.assertEqual(texts, ['ok'])

	def test_invoke_stream_skips_empty_text(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': ''}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke')
		self.assertEqual(texts, ['ok'])

	def test_converse_stream_yields_text(self):
		events = [
			{'contentBlockDelta': {'delta': {'text': 'hello'}}},
			{'contentBlockDelta': {'delta': {'text': ' world'}}},
			{'messageStop': {'stopReason': 'done'}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='converse')
		self.assertEqual(texts, ['hello', ' world'])

	def test_converse_stream_snake_case(self):
		events = [
			{'content_block_delta': {'delta': {'text': 'snake'}}},
			{'message_stop': {'stopReason': 'done'}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='converse')
		self.assertEqual(texts, ['snake'])

	def test_converse_stream_stops_on_message_stop(self):
		events = [
			{'contentBlockDelta': {'delta': {'text': 'a'}}},
			{'messageStop': {'stopReason': 'end_turn'}},
			{'contentBlockDelta': {'delta': {'text': 'ignored'}}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='converse')
		self.assertEqual(texts, ['a'])

	def test_converse_stream_stops_on_custom_key(self):
		events = [
			{'contentBlockDelta': {'delta': {'text': 'a'}}},
			{'stop': True},
			{'contentBlockDelta': {'delta': {'text': 'ignored'}}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='converse', converse_stop_keys=('stop',))
		self.assertEqual(texts, ['a'])

	def test_converse_stream_skips_non_dict_delta(self):
		events = [
			{'contentBlockDelta': 'not-a-dict'},
			{'contentBlockDelta': {'delta': {'text': 'ok'}}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='converse')
		self.assertEqual(texts, ['ok'])

	def test_auto_mode_prefers_converse(self):
		events = [
			{'contentBlockDelta': {'delta': {'text': 'converse'}}},
			{'messageStop': {}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='auto')
		self.assertEqual(texts, ['converse'])

	def test_auto_mode_falls_back_to_invoke(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'invoke'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='auto')
		self.assertEqual(texts, ['invoke'])

	def test_auto_mode_skips_unrecognized(self):
		events = [
			{'unknown': 'event'},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='auto')
		self.assertEqual(texts, ['ok'])

	def test_auto_mode_detects_converse_from_stop_key(self):
		events = [
			{'end': True},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='auto')
		self.assertEqual(texts, [])

	def test_skips_non_dict_events(self):
		events = [
			'not-a-dict',
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}}).encode('utf-8')}},
		]
		texts = []
		iter_bedrock_stream_text(events, on_text=texts.append, stream_kind='invoke')
		self.assertEqual(texts, ['ok'])


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

	def test_exception_returns_empty(self):
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


class TestIterStreamWithCallback(TestCase):
	def test_yields_text(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'hello'}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': ' world'}, 'type': 'end'}).encode('utf-8')}},
		]
		result = list(_iter_stream_with_callback(events, stream_kind='invoke'))
		self.assertEqual(result, ['hello', ' world'])

	def test_closes_stream(self):
		class MockStream:
			def __init__(self):
				self.closed = False

			def __iter__(self):
				yield {'chunk': {'bytes': json.dumps({'delta': {'text': 'hi'}, 'type': 'end'}).encode('utf-8')}}

			def close(self):
				self.closed = True

		stream = MockStream()
		list(_iter_stream_with_callback(stream, stream_kind='invoke'))
		self.assertTrue(stream.closed)

	def test_propagates_exception(self):
		def bad_stream():
			yield {'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}}).encode('utf-8')}}
			raise RuntimeError('stream error')

		gen = _iter_stream_with_callback(bad_stream(), stream_kind='invoke')
		self.assertEqual(next(gen), 'ok')

		with self.assertRaises(RuntimeError) as ctx:
			list(gen)
		self.assertEqual(str(ctx.exception), 'stream error')

	def test_no_close_method(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'hi'}, 'type': 'end'}).encode('utf-8')}},
		]
		result = list(_iter_stream_with_callback(events, stream_kind='invoke'))
		self.assertEqual(result, ['hi'])


class TestIterBedrockStreamTextGenWithTail(TestCase):
	def test_invoke_yields_text_and_captures_usage(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'hello'}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'usage': {'inputTokens': 10}, 'type': 'end'}).encode('utf-8')}},
		]
		captured = []
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=captured.append))
		self.assertEqual(result, ['hello'])
		self.assertTrue(any('usage' in c for c in captured))

	def test_converse_yields_text(self):
		events = [
			{'contentBlockDelta': {'delta': {'text': 'conv'}}},
			{'messageStop': {'stopReason': 'done'}},
		]
		captured = []
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=captured.append))
		self.assertEqual(result, ['conv'])

	def test_captures_invocation_metrics(self):
		events = [
			{
				'chunk': {
					'bytes': json.dumps(
						{'delta': {'text': 'hi'}, 'amazon-bedrock-invocationMetrics': {'latency': 123}}
					).encode('utf-8')
				}
			},
			{'chunk': {'bytes': json.dumps({'type': 'end'}).encode('utf-8')}},
		]
		captured = []
		list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=captured.append))
		self.assertTrue(any('amazon-bedrock-invocationMetrics' in c for c in captured))

	def test_captures_invocation_metrics_alt_key(self):
		events = [
			{
				'chunk': {
					'bytes': json.dumps({'delta': {'text': 'x'}, 'invocationMetrics': {'tokens': 5}}).encode('utf-8')
				}
			},
			{'chunk': {'bytes': json.dumps({'type': 'end'}).encode('utf-8')}},
		]
		captured = []
		list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=captured.append))
		self.assertTrue(any('invocationMetrics' in c for c in captured))

	def test_stops_on_message_stop(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'a'}}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'b'}, 'type': 'message_stop'}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ignored'}}).encode('utf-8')}},
		]
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=lambda x: None))
		self.assertEqual(result, ['a', 'b'])

	def test_stops_on_messageStop(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'a'}, 'type': 'messageStop'}).encode('utf-8')}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ignored'}}).encode('utf-8')}},
		]
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=lambda x: None))
		self.assertEqual(result, ['a'])

	def test_stops_on_end_turn(self):
		events = [
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'a'}, 'type': 'end_turn'}).encode('utf-8')}},
		]
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=lambda x: None))
		self.assertEqual(result, ['a'])

	def test_converse_snake_case(self):
		events = [
			{'content_block_delta': {'delta': {'text': 'snake'}}},
			{'message_stop': {'stopReason': 'done'}},
		]
		captured = []
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=captured.append))
		self.assertEqual(result, ['snake'])

	def test_converse_stops_on_message_stop(self):
		events = [
			{'contentBlockDelta': {'delta': {'text': 'a'}}},
			{'messageStop': {}},
			{'contentBlockDelta': {'delta': {'text': 'ignored'}}},
		]
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=lambda x: None))
		self.assertEqual(result, ['a'])

	def test_skips_non_dict_events(self):
		events = [
			'not-a-dict',
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}, 'type': 'end'}).encode('utf-8')}},
		]
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=lambda x: None))
		self.assertEqual(result, ['ok'])

	def test_handles_invalid_json(self):
		events = [
			{'chunk': {'bytes': b'not-json'}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'valid'}, 'type': 'end'}).encode('utf-8')}},
		]
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=lambda x: None))
		self.assertEqual(result, ['valid'])

	def test_handles_empty_chunk(self):
		events = [
			{'chunk': {}},
			{'chunk': {'bytes': json.dumps({'delta': {'text': 'ok'}, 'type': 'end'}).encode('utf-8')}},
		]
		result = list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=lambda x: None))
		self.assertEqual(result, ['ok'])

	def test_captures_message_stop_in_tail(self):
		events = [
			{'messageStop': {'stopReason': 'end_turn'}},
		]
		captured = []
		list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=captured.append))
		self.assertTrue(any('messageStop' in c for c in captured))

	def test_skips_non_dict_tail(self):
		events = [
			{'chunk': {'bytes': json.dumps('not-a-dict').encode('utf-8')}},
		]
		captured = []
		list(iter_bedrock_stream_text_gen_with_tail(events, on_tail=captured.append))
		self.assertEqual(len(captured), 0)
