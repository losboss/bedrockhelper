from unittest import TestCase
import json

from bedrockhelper.utils import (
	iter_bedrock_stream_text,
	normalize_records,
	truncate_by_chars,
	format_context_passages,
	normalize_headers, extract_converse_text, build_converse_request
)


class TestIterBedrockStreamText(TestCase):
	def test_invoke_emits_text_and_stops_on_type(self):
		events = [
			{"chunk": {"bytes": json.dumps({"delta": {"text": "hello"}}).encode("utf-8")}},
			{"chunk": {"bytes": json.dumps({"delta": {"text": "world"}, "type": "end"}).encode("utf-8")}},
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ignored"}}).encode("utf-8")}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="invoke")
		self.assertEqual(collected, ["hello", "world"])

	def test_converse_emits_text_and_stops_on_message_stop_dict(self):
		events = [
			{"contentBlockDelta": {"delta": {"text": "a"}}},
			{"messageStop": {"stopReason": "done"}},
			{"contentBlockDelta": {"delta": {"text": "ignored"}}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="converse")
		self.assertEqual(collected, ["a"])

	def test_converse_snake_case_and_truthy_stop_key(self):
		events = [
			{"content_block_delta": {"delta": {"text": "snake"}}},
			{"stop": True},
			{"contentBlockDelta": {"delta": {"text": "ignored"}}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="converse")
		self.assertEqual(collected, ["snake"])

	def test_auto_prefers_converse_and_handles_invoke_after(self):
		events = [
			"not-a-dict",
			{
				"contentBlockDelta": {"delta": {"text": "conv"}},
				"chunk": {"bytes": json.dumps({"delta": {"text": "invoke"}}).encode("utf-8")},
			},
			{"chunk": {"bytes": json.dumps({"delta": {"text": "inv2"}, "type": "end"}).encode("utf-8")}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="auto")
		# first event skipped, second handled as converse (conv), third handled as invoke and stops
		self.assertEqual(collected, ["conv", "inv2"])

	def test_invalid_json_invoke_skipped(self):
		events = [
			{"chunk": {"bytes": b"not-json"}},
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ok"}, "type": "end"}).encode("utf-8")}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="invoke")
		self.assertEqual(collected, ["ok"])

	def test_invoke_skips_falsy_chunk_and_processes_following_invoke_event(self):
		events = [
			{"chunk": {}},  # falsy chunk should be ignored
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ok"}, "type": "end"}).encode("utf-8")}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="invoke")
		self.assertEqual(collected, ["ok"])

	def test_invoke_skips_empty_bytes_and_processes_following_invoke_event(self):
		events = [
			{"chunk": {"bytes": b""}},  # empty bytes should be ignored
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ok"}, "type": "end"}).encode("utf-8")}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="invoke")
		self.assertEqual(collected, ["ok"])

	def test_converse_stops_on_message_stop_snake_case_dict(self):
		events = [
			{"contentBlockDelta": {"delta": {"text": "a"}}},
			{"message_stop": {"stopReason": "done"}},  # snake_case dict stop should trigger branch
			{"contentBlockDelta": {"delta": {"text": "ignored"}}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="converse")
		self.assertEqual(collected, ["a"])

	def test_invoke_skips_empty_or_nonstring_text_then_emits_later_and_stops(self):
		"""
		Covers invoke handler branch where delta.text is "" or not a str
		(i.e. the guard `isinstance(t, str) and t` is False).
		"""
		events = [
			# empty string -> should NOT emit
			{"chunk": {"bytes": json.dumps({"delta": {"text": ""}}).encode("utf-8")}},
			# non-string -> should NOT emit
			{"chunk": {"bytes": json.dumps({"delta": {"text": 123}}).encode("utf-8")}},
			# valid -> should emit + stop
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ok"}, "type": "end"}).encode("utf-8")}},
			# would be ignored due to stop
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ignored"}}).encode("utf-8")}},
		]

		collected = []
		iter_bedrock_stream_text(
			events,
			on_text=lambda t: collected.append(t),
			stream_kind="invoke",
			invoke_stop_types=("end",),  # make stop explicit
		)
		self.assertEqual(collected, ["ok"])

	def test_converse_skips_empty_or_nonstring_text_and_stops_via_stop_key(self):
		"""
		Covers converse handler branch where delta.text is "" or not a str,
		and covers stop-key loop by injecting a known stop key.
		"""
		events = [
			# empty string -> should NOT emit
			{"contentBlockDelta": {"delta": {"text": ""}}},
			# non-string -> should NOT emit
			{"contentBlockDelta": {"delta": {"text": 999}}},
			# valid -> should emit
			{"contentBlockDelta": {"delta": {"text": "hi"}}},
			# stop via stop-key loop (NOT messageStop dict)
			{"stop": True},
			# should be ignored due to stop
			{"contentBlockDelta": {"delta": {"text": "ignored"}}},
		]

		collected = []
		iter_bedrock_stream_text(
			events,
			on_text=lambda t: collected.append(t),
			stream_kind="converse",
			converse_stop_keys=("stop",),  # force the stop-key loop branch
		)
		self.assertEqual(collected, ["hi"])

	def test_auto_skips_unrecognized_dict_event_then_processes_invoke(self):
		"""
		Covers auto-mode 'nothing recognizable' branch:
		- dict event with no converse markers, no stop keys, and no chunk -> continue
		Then verifies later invoke event is processed.
		"""
		events = [
			{"foo": "bar"},  # unrecognized in auto mode -> should hit the "skip" else/continue branch
			{"chunk": {"bytes": json.dumps({"delta": {"text": "x"}, "type": "end"}).encode("utf-8")}},
		]

		collected = []
		iter_bedrock_stream_text(
			events,
			on_text=lambda t: collected.append(t),
			stream_kind="auto",
			invoke_stop_types=("end",),
			# keep converse_stop_keys default or set to something that doesn't exist in the first event
			converse_stop_keys=("stop", "stopReason"),  # doesn't matter; first event has neither truthy
		)
		self.assertEqual(collected, ["x"])

	def test_auto_prefers_converse_and_stops_on_converse_stop(self):
		events = [
			{"contentBlockDelta": {"delta": {"text": "conv"}}},
			{"messageStop": {"stopReason": "done"},
			 "chunk": {"bytes": json.dumps({"delta": {"text": "invoke"}, "type": "end"}).encode("utf-8")}},
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ignored"}, "type": "end"}).encode("utf-8")}},
		]
		collected = []
		iter_bedrock_stream_text(events, on_text=lambda t: collected.append(t), stream_kind="auto")
		self.assertEqual(collected, ["conv"])

	def test_auto_routes_to_invoke_and_emits_text_then_stops(self):
		"""
		Forces auto-mode to treat an event as INVOKE (chunk present, no converse signals),
		and ensures the invoke handler's 'emit text' line is executed.
		This tends to cover the stubborn invoke-side uncovered lines (56-57).
		"""
		events = [
			# auto should decide "invoke" because chunk exists and there are no converse markers
			{"chunk": {"bytes": json.dumps({"delta": {"text": "hello"}}).encode("utf-8")}},
			# then stop
			{"chunk": {"bytes": json.dumps({"delta": {"text": "world"}, "type": "end"}).encode("utf-8")}},
		]

		collected = []
		iter_bedrock_stream_text(
			events,
			on_text=lambda t: collected.append(t),
			stream_kind="auto",
			invoke_stop_types=("end",),  # make stop deterministic regardless of defaults
		)
		self.assertEqual(collected, ["hello", "world"])

	def test_converse_emits_text_from_snake_case_content_block_delta(self):
		"""
		Forces the converse handler to use the snake_case key
		`content_block_delta`, not `contentBlockDelta`.
		This commonly covers the stubborn converse-side uncovered lines (80-81).
		"""
		events = [
			{"content_block_delta": {"delta": {"text": "snake"}}},
			{"message_stop": {"stopReason": "done"}},
		]

		collected = []
		iter_bedrock_stream_text(
			events,
			on_text=lambda t: collected.append(t),
			stream_kind="converse",
		)
		self.assertEqual(collected, ["snake"])

	def test_converse_ms_dict_stop_triggers(self):
		events = [
			{"contentBlockDelta": {"delta": {"text": "a"}}},
			{"messageStop": {"stopReason": "done"}},  # <-- ms dict triggers stop
			{"contentBlockDelta": {"delta": {"text": "ignored"}}},
		]
		out = []
		iter_bedrock_stream_text(events, on_text=out.append, stream_kind="converse", converse_stop_keys=())
		assert out == ["a"]

	def test_auto_sets_has_converse_via_stop_key_loop_when_no_converse_shape_keys(self):
		# Use a custom stop key list so we can guarantee which key triggers the branch
		stop_key = "stopKeyX"

		events = [
			# This event has NONE of the "shape" keys:
			# - no contentBlockDelta/content_block_delta
			# - no messageStop/message_stop
			#
			# But it DOES have a truthy stop key, so the else-loop should set has_converse=True and break.
			{stop_key: True},

			# This would be treated as invoke if we ever got here, but we should stop before it.
			{"chunk": {"bytes": json.dumps({"delta": {"text": "ignored"}, "type": "end"}).encode("utf-8")}},
		]

		collected = []
		iter_bedrock_stream_text(
			events,
			on_text=collected.append,
			stream_kind="auto",
			converse_stop_keys=(stop_key,),  # <-- forces the else-loop path
		)

		# No text emitted (we never had a contentBlockDelta), but importantly:
		# we should have stopped on the first event due to the stop key,
		# proving the else-loop ran and broke early.
		self.assertEqual(collected, [])


class TestNormalizeHeaders(TestCase):
	def test_lowercases_keys(self):
		headers = {"Content-Type": "application/json", "X-Amzn-Header": "val"}
		expected = {"content-type": "application/json", "x-amzn-header": "val"}
		self.assertEqual(normalize_headers(headers), expected)

	def test_stringifies_non_str_keys(self):
		headers = {1: "one", None: "none", "True": "bool"}
		expected = {"1": "one", "none": "none", "true": "bool"}
		self.assertEqual(normalize_headers(headers), expected)

	def test_preserves_values_and_types(self):
		complex_value = {"a": 1}
		headers = {"X-Count": 5, b"Binary": complex_value}
		result = normalize_headers(headers)
		self.assertIn("x-count", result)
		self.assertIn("b'binary'", result)  # bytes key stringified
		self.assertEqual(result["x-count"], 5)
		self.assertIs(result["b'binary'"], complex_value)

	def test_empty_input_returns_empty_dict(self):
		self.assertEqual(normalize_headers({}), {})

	def test_accepts_mapping_like_object(self):
		class DummyMap:
			def items(self):
				return [("Some-Key", "v")]

		dm = DummyMap()
		self.assertEqual(normalize_headers(dm), {"some-key": "v"})


class TestFormatContextPassages(TestCase):
	def test_no_headers_default(self):
		passages = ["one", "two"]
		expected = "one\n\ntwo"
		self.assertEqual(format_context_passages(passages), expected)

	def test_with_headers_default_template(self):
		passages = ["a", "b"]
		expected = "--- passage 1 ---\na\n\n--- passage 2 ---\nb"
		self.assertEqual(format_context_passages(passages, include_headers=True), expected)

	def test_custom_header_template(self):
		passages = ["x"]
		expected = "### passage_01 ###\nx"
		result = format_context_passages(
			passages,
			include_headers=True,
			header_template="### passage_{i:02d} ###"
		)
		self.assertEqual(result, expected)

	def test_custom_separator(self):
		passages = ["p1", "p2"]
		expected = "p1\n---\np2"
		self.assertEqual(format_context_passages(passages, separator="\n---\n"), expected)

	def test_non_string_converted(self):
		passages = [1, None, True]
		expected = "1\n\nNone\n\nTrue"
		self.assertEqual(format_context_passages(passages), expected)

	def test_empty_passages(self):
		self.assertEqual(format_context_passages([]), "")


class TestTruncateByChars(TestCase):
	def test_returns_original_when_max_none(self):
		self.assertEqual(truncate_by_chars("hello", None), "hello")

	def test_returns_empty_when_max_nonpositive(self):
		self.assertEqual(truncate_by_chars("hello", 0), "")
		self.assertEqual(truncate_by_chars("hello", -3), "")

	def test_truncates_correctly(self):
		self.assertEqual(truncate_by_chars("abcdef", 3), "abc")
		self.assertEqual(truncate_by_chars("short", 10), "short")


class TestNormalizeRecords(TestCase):
	def test_normalize_records_handles_dict_input(self):
		records = {"key1": "value1", "key2": "value2"}
		result = normalize_records(records)
		assert result == [("key1", "value1"), ("key2", "value2")]


	def test_normalize_records_handles_list_of_tuples(self):
		records = [("key1", "value1"), ("key2", "value2")]
		result = normalize_records(records)
		assert result == [("key1", "value1"), ("key2", "value2")]


	def test_normalize_records_handles_list_of_mappings(self):
		records = [{"key1": "value1"}, {"key2": "value2"}]
		result = normalize_records(records)
		assert result == [("key1", "value1"), ("key2", "value2")]


	def test_normalize_records_raises_error_for_invalid_mapping(self):
		records = [{"key1": "value1", "key2": "value2"}]
		try:
			normalize_records(records)
			assert False, "Expected ValueError"
		except ValueError as e:
			assert str(e) == "Each mapping record must have exactly 1 entry; got 2"


	def test_normalize_records_raises_error_for_unsupported_type(self):
		records = ["unsupported"]
		try:
			normalize_records(records)
			assert False, "Expected ValueError"
		except ValueError as e:
			assert str(e) == "Unsupported record item type: <class 'str'>"


	def test_normalize_records_handles_empty_input(self):
		records = []
		result = normalize_records(records)
		assert result == []


class TestExtractConverseText(TestCase):
	def test_extracts_text_from_multiple_parts(self):
		resp = {"output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}}}
		self.assertEqual(extract_converse_text(resp), "hello world")

	def test_ignores_non_dict_and_non_string_items(self):
		resp = {"output": {"message": {"content": ["skip", {"text": 123}, {"text": "ok"}]}}}
		self.assertEqual(extract_converse_text(resp), "ok")

	def test_missing_keys_return_empty_string(self):
		self.assertEqual(extract_converse_text({}), "")
		self.assertEqual(extract_converse_text({"output": {}}), "")
		self.assertEqual(extract_converse_text({"output": {"message": {}}}), "")

	def test_none_input_returns_empty_string(self):
		self.assertEqual(extract_converse_text(None), "")

	def test_empty_content_returns_empty_string(self):
		resp = {"output": {"message": {"content": []}}}
		self.assertEqual(extract_converse_text(resp), "")


class TestBuildConverseRequest(TestCase):
	def test_builds_required_fields(self):
		req = build_converse_request(
			model_id="m1",
			system_prompt="sys",
			context="ctx",
			question="q",
			max_tokens=50,
			temperature=0.7,
		)
		self.assertEqual(req["modelId"], "m1")
		self.assertEqual(req["system"], [{"text": "sys"}])
		self.assertIn("messages", req)
		self.assertIsInstance(req["messages"], list)
		# single user message with content->text present
		msg = req["messages"][0]
		self.assertEqual(msg["role"], "user")
		self.assertIn("content", msg)
		self.assertEqual(msg["content"][0]["text"], "Context:\nctx\n\nQuestion:\nq\n")
		self.assertIn("inferenceConfig", req)
		self.assertEqual(req["inferenceConfig"]["maxTokens"], 50)
		self.assertEqual(req["inferenceConfig"]["temperature"], 0.7)

	def test_includes_rag_instructions_when_provided(self):
		req = build_converse_request(
			model_id="m2",
			system_prompt="s",
			context="C",
			question="Q",
			max_tokens=10,
			temperature=0.1,
			rag_instructions="USE_RAG"
		)
		text = req["messages"][0]["content"][0]["text"]
		# rag_instructions should be prepended followed by a newline
		self.assertTrue(text.startswith("USE_RAG\nContext:\nC\n\nQuestion:\nQ\n"))

	def test_extra_params_override_and_add_fields(self):
		req = build_converse_request(
			model_id="original",
			system_prompt="s",
			context="c",
			question="q",
			max_tokens=5,
			temperature=0.2,
			inferenceConfig={"maxTokens": 1, "temperature": 0.05},
			modelId="overridden",  # intentional collision to ensure update wins
			extra_top_level="value",
		)
		# extra top-level keys should be present
		self.assertEqual(req["extra_top_level"], "value")
		# update should have allowed overriding modelId
		self.assertEqual(req["modelId"], "overridden")
		# the provided inferenceConfig should replace the default (no deep merge)
		self.assertEqual(req["inferenceConfig"], {"maxTokens": 1, "temperature": 0.05})

	def test_preserves_types_and_format_of_user_text(self):
		req = build_converse_request(
			model_id="x",
			system_prompt="sys",
			context=123,  # non-string context should be stringified in user_text
			question=None,
			max_tokens=0,
			temperature=0.0,
		)
		expected_text = "Context:\n123\n\nQuestion:\nNone\n"
		self.assertEqual(req["messages"][0]["content"][0]["text"], expected_text)
		# inferenceConfig types preserved as passed
		self.assertEqual(req["inferenceConfig"]["maxTokens"], 0)
		self.assertEqual(req["inferenceConfig"]["temperature"], 0.0)
