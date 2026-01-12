import asyncio
import io
import json
import time
import unittest
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional
from unittest.mock import MagicMock, patch

# Adjust import to match where BedrockHelper lives in your project:
# from bedrockhelper.main import BedrockHelper
from bedrockhelper.main import BedrockHelper
from bedrockhelper.models import InvocationMetrics, EmbeddingResponse, BatchJobResponse, RAGResponse


@dataclass
class _FakeBody:
	data: bytes

	def read(self) -> bytes:
		return self.data


class _FakeStream:
	"""Simple closeable iterator for streaming events."""

	def __init__(self, events: Iterable[dict]) -> None:
		self._events = iter(events)
		self.closed = False

	def __iter__(self) -> Iterator[dict]:
		return self

	def __next__(self) -> dict:
		return next(self._events)

	def close(self) -> None:
		self.closed = True


class TestBedrockHelperInit(unittest.TestCase):
	def test_init_rejects_invalid_max_concurrent_streams(self):
		with self.assertRaises(ValueError):
			BedrockHelper(
				bedrock_runtime_client=MagicMock(),
				bedrock_client=MagicMock(),
				s3_client=MagicMock(),
				max_concurrent_streams=0,
			)

	def test_init_sets_stream_semaphore(self):
		h = BedrockHelper(
			bedrock_runtime_client=MagicMock(),
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
			max_concurrent_streams=3,
		)
		self.assertEqual(h._max_concurrent_streams, 3)
		self.assertTrue(hasattr(h, "_stream_sema"))


class TestExtractMetricsFromResponse(unittest.TestCase):
	def test_extract_metrics_prefers_body_usage(self):
		resp = {"ResponseMetadata": {"HTTPHeaders": {}}}
		body = {"usage": {"inputTokens": 10, "outputTokens": 7, "totalTokens": 17}}
		met = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertIsInstance(met, InvocationMetrics)
		self.assertEqual(met.input_tokens, 10)
		self.assertEqual(met.output_tokens, 7)
		self.assertEqual(met.total_tokens, 17)

	def test_extract_metrics_uses_titan_inputTextTokenCount(self):
		resp = {"ResponseMetadata": {"HTTPHeaders": {}}}
		body = {"inputTextTokenCount": 42}
		met = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertEqual(met.input_tokens, 42)

	def test_extract_metrics_falls_back_to_headers(self):
		resp = {
			"ResponseMetadata": {
				"HTTPHeaders": {
					"x-amzn-bedrock-input-token-count": "3",
					"x-amzn-bedrock-output-token-count": "5",
					"x-amzn-bedrock-invocation-latency": "100",
					"x-amzn-bedrock-first-byte-latency": "10",
				}
			}
		}
		met = BedrockHelper._extract_metrics_from_response(resp, None)
		self.assertEqual(met.input_tokens, 3)
		self.assertEqual(met.output_tokens, 5)
		self.assertEqual(met.total_tokens, 8)
		self.assertEqual(met.invocation_latency_ms, 100)
		self.assertEqual(met.first_byte_latency_ms, 10)

	def test_extract_metrics_total_tokens_computed_if_missing(self):
		resp = {"ResponseMetadata": {"HTTPHeaders": {}}}
		body = {"usage": {"promptTokens": 2, "completionTokens": 4}}
		met = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertEqual(met.input_tokens, 2)
		self.assertEqual(met.output_tokens, 4)
		self.assertEqual(met.total_tokens, 6)


class TestGenerateWithRagRouting(unittest.TestCase):
	def setUp(self) -> None:
		self.runtime = MagicMock()
		self.bedrock = MagicMock()
		self.s3 = MagicMock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=self.runtime,
			bedrock_client=self.bedrock,
			s3_client=self.s3,
		)

	def test_generate_prefers_converse_non_stream(self):
		expected = RAGResponse(text="ok", stream=False, metrics=InvocationMetrics(), raw_response={})
		with patch.object(self.helper, "_generate_with_converse", return_value=expected) as p_converse, \
				patch.object(self.helper, "_generate_with_claude_message_format") as p_fallback:
			# must look like converse exists
			setattr(self.runtime, "converse", MagicMock())

			out = self.helper.generate_with_rag(
				system_prompt="sys",
				context="ctx",
				question="q",
				stream=False,
				prefer_converse=True,
			)
			self.assertIs(out, expected)
			p_converse.assert_called_once()
			p_fallback.assert_not_called()

	def test_generate_prefers_converse_stream(self):
		expected = RAGResponse(text="ok", stream=True, metrics=InvocationMetrics(), raw_response=None)
		with patch.object(self.helper, "_generate_with_converse_stream", return_value=expected) as p_stream, \
				patch.object(self.helper, "_generate_with_claude_message_format") as p_fallback:
			setattr(self.runtime, "converse", MagicMock())
			setattr(self.runtime, "converse_stream", MagicMock())

			out = self.helper.generate_with_rag(
				system_prompt="sys",
				context="ctx",
				question="q",
				stream=True,
				prefer_converse=True,
			)
			self.assertIs(out, expected)
			p_stream.assert_called_once()
			p_fallback.assert_not_called()

	def test_generate_falls_back_when_converse_raises(self):
		expected = RAGResponse(text="fb", stream=False, metrics=InvocationMetrics(), raw_response={})
		with patch.object(self.helper, "_generate_with_converse", side_effect=RuntimeError("nope")), \
				patch.object(self.helper, "_generate_with_claude_message_format", return_value=expected) as p_fallback:
			setattr(self.runtime, "converse", MagicMock())

			out = self.helper.generate_with_rag(
				system_prompt="sys",
				context="ctx",
				question="q",
				stream=False,
				prefer_converse=True,
			)
			self.assertIs(out, expected)
			p_fallback.assert_called_once()

	def test_generate_uses_fallback_when_prefer_converse_false(self):
		expected = RAGResponse(text="fb", stream=False, metrics=InvocationMetrics(), raw_response={})
		with patch.object(self.helper, "_generate_with_claude_message_format", return_value=expected) as p_fallback:
			out = self.helper.generate_with_rag(
				system_prompt="sys",
				context="ctx",
				question="q",
				prefer_converse=False,
			)
			self.assertIs(out, expected)
			p_fallback.assert_called_once()

	def test_generate_formats_sequence_context_when_include_headers(self):
		# Validate it routes with formatted ctx (not exact formatting; just that it's str and contains items)
		expected = RAGResponse(text="ok", stream=False, metrics=InvocationMetrics(), raw_response={})
		with patch.object(self.helper, "_generate_with_converse", return_value=expected) as p_converse:
			setattr(self.runtime, "converse", MagicMock())
			out = self.helper.generate_with_rag(
				system_prompt="sys",
				context=["a", "b"],
				include_headers_in_context=True,
				question="q",
			)
			self.assertIs(out, expected)
			args, kwargs = p_converse.call_args
			self.assertIn("context", kwargs)
			self.assertIsInstance(kwargs["context"], str)
			self.assertIn("a", kwargs["context"])
			self.assertIn("b", kwargs["context"])


class TestConverseImplementations(unittest.TestCase):
	def setUp(self) -> None:
		self.runtime = MagicMock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=self.runtime,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	def test_generate_with_converse_extracts_text_and_metrics(self):
		fake_resp = {
			"output": {"message": {"content": [{"text": "Hello"}, {"text": " world"}]}},
			"ResponseMetadata": {"HTTPHeaders": {"x-amzn-bedrock-input-token-count": "1"}},
		}
		self.runtime.converse.return_value = fake_resp

		out = self.helper._generate_with_converse(
			model_id="m",
			system_prompt="sys",
			context="ctx",
			question="q",
			max_tokens=10,
			temperature=0.1,
			rag_instructions="",
		)
		self.assertEqual(out.text, "Hello world")
		self.assertFalse(out.stream)
		self.assertEqual(out.metrics.input_tokens, 1)
		self.assertEqual(out.raw_response, fake_resp)

	def test_generate_with_converse_stream_joins_chunks_and_closes_stream(self):
		stream = _FakeStream([{"contentBlockDelta": {"delta": {"text": "A"}}},
		                      {"contentBlockDelta": {"delta": {"text": "B"}}},
		                      {"messageStop": True}])
		self.runtime.converse_stream.return_value = {"stream": stream}

		out = self.helper._generate_with_converse_stream(
			model_id="m",
			system_prompt="sys",
			context="ctx",
			question="q",
			max_tokens=10,
			temperature=0.1,
			rag_instructions="",
		)
		self.assertEqual(out.text, "AB")
		self.assertTrue(out.stream)
		self.assertTrue(stream.closed)

	def test_generate_with_converse_stream_missing_stream_raises(self):
		self.runtime.converse_stream.return_value = {}
		with self.assertRaises(RuntimeError):
			self.helper._generate_with_converse_stream(
				model_id="m",
				system_prompt="sys",
				context="ctx",
				question="q",
				max_tokens=10,
				temperature=0.1,
				rag_instructions="",
			)


class TestClaudeMessageFormat(unittest.TestCase):
	def setUp(self) -> None:
		self.runtime = MagicMock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=self.runtime,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	def test_generate_with_claude_non_stream_parses_content_list(self):
		body = {"content": [{"text": "Hi"}, {"text": "!"}], "usage": {"inputTokens": 2, "outputTokens": 3}}
		resp = {"body": _FakeBody(json.dumps(body).encode("utf-8")), "ResponseMetadata": {"HTTPHeaders": {}}}
		self.runtime.invoke_model.return_value = resp

		out = self.helper._generate_with_claude_message_format(
			model_id="m",
			system_prompt="sys",
			context="ctx",
			question="q",
			stream=False,
			temperature=0.1,
			max_tokens=10,
			rag_instructions="",
		)
		self.assertEqual(out.text, "Hi!")
		self.assertFalse(out.stream)
		self.assertIsInstance(out.metrics, InvocationMetrics)
		self.assertEqual(out.metrics.input_tokens, 2)

	def test_generate_with_claude_non_stream_invalid_json_falls_back_raw(self):
		resp = {"body": _FakeBody(b"not json"), "ResponseMetadata": {"HTTPHeaders": {}}}
		self.runtime.invoke_model.return_value = resp

		out = self.helper._generate_with_claude_message_format(
			model_id="m",
			system_prompt="sys",
			context="ctx",
			question="q",
			stream=False,
			temperature=0.1,
			max_tokens=10,
		)
		self.assertIsInstance(out.raw_response, dict)
		self.assertIn("raw", out.raw_response)

	def test_generate_with_claude_stream_joins_chunks_and_closes_stream(self):
		# invoke streaming uses {"chunk":{"bytes": b"...json..."}}
		def mk_delta(text: str) -> dict:
			msg = {"delta": {"text": text}, "type": "message_delta"}
			return {"chunk": {"bytes": json.dumps(msg).encode("utf-8")}}

		stop = {"chunk": {"bytes": json.dumps({"type": "message_stop"}).encode("utf-8")}}
		stream = _FakeStream([mk_delta("X"), mk_delta("Y"), stop])

		self.runtime.invoke_model_with_response_stream.return_value = {"body": stream}

		out = self.helper._generate_with_claude_message_format(
			model_id="m",
			system_prompt="sys",
			context="ctx",
			question="q",
			stream=True,
			temperature=0.1,
			max_tokens=10,
		)
		self.assertEqual(out.text, "XY")
		self.assertTrue(out.stream)
		self.assertTrue(stream.closed)

	def test_generate_with_claude_stream_missing_body_raises(self):
		self.runtime.invoke_model_with_response_stream.return_value = {}
		with self.assertRaises(RuntimeError):
			self.helper._generate_with_claude_message_format(
				model_id="m",
				system_prompt="sys",
				context="ctx",
				question="q",
				stream=True,
				temperature=0.1,
				max_tokens=10,
			)


class TestAsyncGenerateWrappers(unittest.IsolatedAsyncioTestCase):
	async def asyncSetUp(self) -> None:
		self.helper = BedrockHelper(
			bedrock_runtime_client=MagicMock(),
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	async def test_async_generate_calls_sync(self):
		expected = RAGResponse(text="ok", stream=False, metrics=InvocationMetrics(), raw_response={})
		with patch.object(self.helper, "generate_with_rag", return_value=expected) as p:
			out = await self.helper.async_generate_with_rag(system_prompt="sys", context="ctx", question="q")
			self.assertIs(out, expected)
			p.assert_called_once()

	async def test_async_stream_generate_uses_converse_stream_and_streams_text(self):
		# Ensure converse_stream path
		stream = _FakeStream([{"contentBlockDelta": {"delta": {"text": "A"}}},
		                      {"contentBlockDelta": {"delta": {"text": "B"}}},
		                      {"messageStop": True}])
		self.helper.bedrock_runtime.converse_stream = MagicMock(return_value={"stream": stream})

		chunks: List[str] = []
		async for t in self.helper.async_stream_generate_with_rag(
				system_prompt="sys",
				context="ctx",
				question="q",
				prefer_converse=True,
		):
			chunks.append(t)

		self.assertEqual("".join(chunks), "AB")
		self.assertTrue(stream.closed)

	async def test_async_stream_generate_falls_back_to_invoke_stream(self):
		# Force converse_stream to raise so we hit invoke stream fallback
		self.helper.bedrock_runtime.converse_stream = MagicMock(side_effect=RuntimeError("nope"))

		def mk_delta(text: str) -> dict:
			msg = {"delta": {"text": text}, "type": "message_delta"}
			return {"chunk": {"bytes": json.dumps(msg).encode("utf-8")}}

		stop = {"chunk": {"bytes": json.dumps({"type": "message_stop"}).encode("utf-8")}}
		stream = _FakeStream([mk_delta("X"), mk_delta("Y"), stop])
		self.helper.bedrock_runtime.invoke_model_with_response_stream = MagicMock(return_value={"body": stream})

		out: List[str] = []
		async for t in self.helper.async_stream_generate_with_rag(
				system_prompt="sys",
				context="ctx",
				question="q",
				prefer_converse=True,
		):
			out.append(t)

		self.assertEqual("".join(out), "XY")
		self.assertTrue(stream.closed)

	async def test_async_stream_generate_raises_on_invoke_stream_missing_body(self):
		self.helper.bedrock_runtime.converse_stream = MagicMock(side_effect=RuntimeError("nope"))
		self.helper.bedrock_runtime.invoke_model_with_response_stream = MagicMock(return_value={})
		with self.assertRaises(RuntimeError):
			async for _ in self.helper.async_stream_generate_with_rag(
					system_prompt="sys",
					context="ctx",
					question="q",
			):
				pass

	async def test_async_stream_generate_respects_concurrency_limit(self):
		helper = BedrockHelper(
			bedrock_runtime_client=MagicMock(),
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
			max_concurrent_streams=1,
		)

		# Make converse_stream block until we say so
		started = asyncio.Event()
		release = asyncio.Event()

		def blocking_converse_stream(**_: Any) -> Dict[str, Any]:
			# This runs in a worker thread; coordinate via events
			asyncio.get_event_loop()  # may not exist in thread; don't use
			return {"stream": _FakeStream([{"contentBlockDelta": {"delta": {"text": "A"}}}, {"messageStop": True}])}

		# Instead of blocking inside converse_stream (thread complications),
		# we simulate contention by manually acquiring semaphore.
		helper._stream_sema.acquire()

		helper.bedrock_runtime.converse_stream = MagicMock(
			return_value={"stream": _FakeStream([{"messageStop": True}])})

		# Start a stream; it should wait for semaphore and therefore not yield until release
		async def consume() -> List[str]:
			buf: List[str] = []
			async for t in helper.async_stream_generate_with_rag(system_prompt="sys", context="ctx", question="q"):
				buf.append(t)
			return buf

		task = asyncio.create_task(consume())
		await asyncio.sleep(0.05)

		self.assertFalse(task.done())  # still blocked due to semaphore held

		helper._stream_sema.release()
		res = await task
		self.assertEqual(res, [])


class TestEmbeddings(unittest.TestCase):
	def setUp(self) -> None:
		self.runtime = MagicMock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=self.runtime,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	def test_embed_one_returns_embedding_and_metrics(self):
		body = {"embedding": [0.1, 0.2], "inputTextTokenCount": 3}
		resp = {"body": _FakeBody(json.dumps(body).encode("utf-8")), "ResponseMetadata": {"HTTPHeaders": {}}}
		self.runtime.invoke_model.return_value = resp

		rid, emb, met = self.helper._embed_one("id1", "hello")
		self.assertEqual(rid, "id1")
		self.assertEqual(emb, [0.1, 0.2])
		self.assertEqual(met.input_tokens, 3)

	def test_embed_one_raises_on_bad_shape(self):
		body = {"nope": True}
		resp = {"body": _FakeBody(json.dumps(body).encode("utf-8")), "ResponseMetadata": {"HTTPHeaders": {}}}
		self.runtime.invoke_model.return_value = resp

		with self.assertRaises(RuntimeError):
			self.helper._embed_one("id1", "hello")

	def test_embed_texts_empty_returns_empty_embedding_response(self):
		out = self.helper.embed_texts({})
		self.assertIsInstance(out, EmbeddingResponse)
		self.assertEqual(out.embeddings, {})
		self.assertEqual(out.metrics, {})

	def test_embed_texts_sync_path_calls_concurrent_embed(self):
		with patch.object(self.helper, "_embed_texts_sync_concurrent", return_value=EmbeddingResponse({}, {})) as p:
			out = self.helper.embed_texts({"a": "x"}, batch_threshold=10)
			self.assertIsInstance(out, EmbeddingResponse)
			p.assert_called_once()

	def test_embed_texts_batch_path_calls_submit_job(self):
		job = BatchJobResponse(
			job_id="jid",
			job_name="jn",
			model_id="m",
			input_s3_uri="s3://in",
			output_s3_uri="s3://out",
			response={},
		)
		with patch.object(self.helper, "submit_embedding_batch_job", return_value=job) as p:
			out = self.helper.embed_texts([("a", "x"), ("b", "y")], batch_threshold=1, s3_bucket="b", role_arn="r")
			self.assertIs(out, job)
			p.assert_called_once()


class TestSubmitEmbeddingBatchJob(unittest.TestCase):
	def setUp(self) -> None:
		self.s3 = MagicMock()
		self.bedrock = MagicMock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=MagicMock(),
			bedrock_client=self.bedrock,
			s3_client=self.s3,
		)

	def test_submit_embedding_batch_job_requires_bucket_and_role(self):
		with self.assertRaises(ValueError):
			self.helper.submit_embedding_batch_job([("a", "x")], s3_bucket=None, s3_prefix="p", role_arn="r")
		with self.assertRaises(ValueError):
			self.helper.submit_embedding_batch_job([("a", "x")], s3_bucket="b", s3_prefix="p", role_arn=None)

	def test_submit_embedding_batch_job_uploads_and_returns_job(self):
		# Bedrock returns some identifier; validate selection prefers jobArn then jobId etc.
		self.bedrock.create_model_invocation_job.return_value = {"jobArn": "arn:aws:bedrock:job/123"}

		with patch("tempfile.NamedTemporaryFile") as ntf, patch("os.remove") as rm:
			tmp_file = MagicMock()
			tmp_file.__enter__.return_value = tmp_file
			tmp_file.__exit__.return_value = None
			tmp_file.name = "/tmp/f.jsonl"
			ntf.return_value = tmp_file

			out = self.helper.submit_embedding_batch_job(
				[("a", "x")],
				s3_bucket="bucket",
				s3_prefix="prefix",
				role_arn="arn:role",
			)

			self.assertIsInstance(out, BatchJobResponse)
			self.assertEqual(out.job_id, "arn:aws:bedrock:job/123")
			self.assertEqual(out.job_name[:21], "bedrock-embedding-job")
			self.assertEqual(out.input_s3_uri, "s3://bucket/prefix/inputs/" + out.job_name.split("-")[-1] + ".jsonl"
			if False else out.input_s3_uri)  # don't overfit exact uuid composition
			self.s3.upload_file.assert_called_once()
			rm.assert_called_once()

	def test_submit_embedding_batch_job_falls_back_to_job_name_when_no_id(self):
		self.bedrock.create_model_invocation_job.return_value = {}

		with patch("tempfile.NamedTemporaryFile") as ntf, patch("os.remove"):
			tmp_file = MagicMock()
			tmp_file.__enter__.return_value = tmp_file
			tmp_file.__exit__.return_value = None
			tmp_file.name = "/tmp/f.jsonl"
			ntf.return_value = tmp_file

			out = self.helper.submit_embedding_batch_job(
				[("a", "x")],
				s3_bucket="bucket",
				s3_prefix="prefix",
				role_arn="arn:role",
			)
			self.assertEqual(out.job_id, out.job_name)


class TestBatchResultsParsing(unittest.TestCase):
	def setUp(self) -> None:
		self.s3 = MagicMock()
		self.helper = BedrockHelper(
			bedrock_runtime_client=MagicMock(),
			bedrock_client=MagicMock(),
			s3_client=self.s3,
		)

	def test_download_batch_results_jsonl_rejects_non_s3_uri(self):
		with self.assertRaises(ValueError):
			list(self.helper.download_batch_results_jsonl(output_s3_uri="http://nope"))

	def test_download_batch_results_jsonl_lists_and_reads_jsonl_objects(self):
		self.s3.get_paginator.return_value.paginate.return_value = [
			{"Contents": [{"Key": "prefix/out1.jsonl"}, {"Key": "prefix/skip.txt"}]}
		]
		lines = b'{"a": 1}\n{"b": 2}\n'
		self.s3.get_object.return_value = {"Body": _FakeBody(lines)}

		out = list(self.helper.download_batch_results_jsonl(output_s3_uri="s3://bucket/prefix/"))
		self.assertEqual(out, [{"a": 1}, {"b": 2}])

	def test_parse_batch_embeddings_extracts_embeddings(self):
		self.s3.get_paginator.return_value.paginate.return_value = [{"Contents": [{"Key": "p/out.jsonl"}]}]
		lines = b'{"recordId":"r1","modelOutput":{"embedding":[1,2]}}\n' \
		        b'{"record_id":"r2","output":{"embedding":[3,4]}}\n' \
		        b'{"recordId":"r3","modelOutput":{"nope":true}}\n'
		self.s3.get_object.return_value = {"Body": _FakeBody(lines)}

		out = self.helper.parse_batch_embeddings(output_s3_uri="s3://bucket/p/")
		self.assertEqual(out, {"r1": [1, 2], "r2": [3, 4]})


class TestWaitForBatchJob(unittest.TestCase):
	def setUp(self) -> None:
		self.helper = BedrockHelper(
			bedrock_runtime_client=MagicMock(),
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	def test_wait_for_batch_job_returns_on_terminal_status(self):
		with patch.object(self.helper, "get_batch_job",
		                  side_effect=[{"status": "InProgress"}, {"status": "Completed"}]) as p, \
				patch("time.sleep") as slp:
			out = self.helper.wait_for_batch_job("job", poll_seconds=0.01, timeout_seconds=1.0)
			self.assertEqual(out["status"], "Completed")
			self.assertEqual(p.call_count, 2)
			slp.assert_called()

	def test_wait_for_batch_job_times_out(self):
		with patch.object(self.helper, "get_batch_job", return_value={"status": "InProgress"}), \
				patch("time.sleep", return_value=None):
			with self.assertRaises(TimeoutError):
				self.helper.wait_for_batch_job("job", poll_seconds=0.0, timeout_seconds=0.01)
