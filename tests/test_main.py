import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator, List
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

from bedrockhelper import RAGResponse
from bedrockhelper.main import BedrockHelper


# -----------------------------
# Test helpers / fakes
# -----------------------------

class FakeStream:
	def __init__(self) -> None:
		self.closed = False

	def close(self) -> None:
		self.closed = True


class FakeBody:
	def __init__(self, payload: bytes) -> None:
		self._payload = payload

	def read(self) -> bytes:
		return self._payload


class FakePaginator:
	def __init__(self, pages: List[dict]) -> None:
		self._pages = pages

	def paginate(self, **kwargs: Any) -> Iterator[dict]:
		yield from self._pages


@dataclass(frozen=True, slots=True)
class DummyEmbeddingJob:
	job_id: str
	job_name: str = "job-name"
	model_id: str = "model-id"
	input_s3_uri: str = "s3://in/x"
	output_s3_uri: str = "s3://out/x"

	def to_ref(self) -> Any:
		return SimpleNamespace(
			job_id=self.job_id,
			job_name=self.job_name,
			model_id=self.model_id,
			input_s3_uri=self.input_s3_uri,
			output_s3_uri=self.output_s3_uri,
		)


class CountingSemaphore:
	"""Drop-in replacement for BoundedSemaphore for unit tests."""

	def __init__(self) -> None:
		self.acquire_calls = 0
		self.release_calls = 0

	def acquire(self) -> bool:
		self.acquire_calls += 1
		return True

	def release(self) -> None:
		self.release_calls += 1


class ReleaseRaisesSemaphore(CountingSemaphore):
	def release(self) -> None:
		super().release()
		raise RuntimeError("release failed")

# ============================================================
# __init__ + metrics tests
# ============================================================

class TestBedrockHelperInitAndMetrics(TestCase):
	def test_init_rejects_invalid_max_concurrent_streams(self):
		with self.assertRaises(ValueError):
			BedrockHelper(
				max_concurrent_streams=0,
				bedrock_runtime_client=MagicMock(),
				bedrock_client=MagicMock(),
				s3_client=MagicMock(),
			)

	def test_init_uses_provided_clients_and_sets_semaphore(self):
		brt = MagicMock()
		br = MagicMock()
		s3 = MagicMock()

		h = BedrockHelper(
			bedrock_runtime_client=brt,
			bedrock_client=br,
			s3_client=s3,
			max_concurrent_streams=3,
		)

		self.assertIs(h.bedrock_runtime, brt)
		self.assertIs(h.bedrock, br)
		self.assertIs(h.s3, s3)
		self.assertEqual(h._max_concurrent_streams, 3)
		self.assertTrue(hasattr(h, "_stream_sema"))

	def test_init_calls_boto3_client_when_clients_not_provided(self):
		with patch("bedrockhelper.main.boto3.client") as pclient:
			pclient.side_effect = [MagicMock(), MagicMock(), MagicMock()]
			h = BedrockHelper(max_concurrent_streams=1)

			self.assertIsNotNone(h.bedrock_runtime)
			self.assertIsNotNone(h.bedrock)
			self.assertIsNotNone(h.s3)
			self.assertEqual(pclient.call_count, 3)

	@patch("bedrockhelper.main.normalize_headers")
	def test_extract_metrics_prefers_usage_then_headers_then_compute_total(self, norm_headers: MagicMock):
		# headers normalized and used only when body didn't provide tokens
		norm_headers.return_value = {
			"x-amzn-bedrock-input-token-count": "11",
			"x-amzn-bedrock-output-token-count": "22",
			"x-amzn-bedrock-invocation-latency": "33",
			"x-amzn-bedrock-first-byte-latency": "44",
		}

		resp = {"ResponseMetadata": {"HTTPHeaders": {"X-AMZN-BEDROCK-INPUT-TOKEN-COUNT": "11"}}}
		body = {
			"usage": {
				# intentionally leave these invalid so headers take over
				"inputTokens": "not-an-int",
				"outputTokens": None,
				"totalTokens": None,
			}
		}

		m = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertEqual(m.input_tokens, 11)
		self.assertEqual(m.output_tokens, 22)
		self.assertEqual(m.total_tokens, 33)  # computed 11+22
		self.assertEqual(m.invocation_latency_ms, 33)
		self.assertEqual(m.first_byte_latency_ms, 44)

	def test_extract_metrics_titan_input_text_token_count_branch(self):
		resp = {"ResponseMetadata": {"HTTPHeaders": {}}}
		body = {"inputTextTokenCount": 99}
		m = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertEqual(m.input_tokens, 99)

	def test_extract_metrics_invocation_metrics_branch(self):
		resp = {"ResponseMetadata": {"HTTPHeaders": {}}}
		body = {
			"amazon-bedrock-invocationMetrics": {
				"invocationLatency": 123,
				"firstByteLatency": 456,
			}
		}
		m = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertEqual(m.invocation_latency_ms, 123)
		self.assertEqual(m.first_byte_latency_ms, 456)

	def test_extract_metrics_handles_no_body_and_no_headers(self):
		m = BedrockHelper._extract_metrics_from_response({}, None)
		# just ensure it returns an InvocationMetrics instance with no crash
		self.assertIsNotNone(m)


# ============================================================
# generate_with_rag routing tests
# ============================================================

class TestBedrockHelperGenerateWithRagRouting(TestCase):
	def setUp(self) -> None:
		self.brt = MagicMock()
		self.br = MagicMock()
		self.s3 = MagicMock()
		self.h = BedrockHelper(
			bedrock_runtime_client=self.brt,
			bedrock_client=self.br,
			s3_client=self.s3,
		)

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	@patch("bedrockhelper.main.format_context_passages", return_value="FORMATTED")
	def test_generate_with_rag_context_list_uses_formatter(self, fmt: MagicMock, trunc: MagicMock):
		self.brt.converse = MagicMock()

		self.h._generate_with_converse = MagicMock(return_value="CONVERSE")
		out = self.h.generate_with_rag(
			system_prompt="SYS",
			context=["a", "b"],
			include_headers_in_context=True,
			question="Q",
			prefer_converse=True,
		)
		self.assertEqual(out, "CONVERSE")
		fmt.assert_called_once()
		_, kwargs = fmt.call_args
		self.assertEqual(kwargs["include_headers"], True)

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	def test_generate_with_rag_prefers_converse_stream_when_stream_true(self, trunc: MagicMock):
		self.brt.converse = MagicMock()
		self.brt.converse_stream = MagicMock()

		self.h._generate_with_converse_stream = MagicMock(return_value="STREAM")
		out = self.h.generate_with_rag(
			system_prompt="SYS",
			context="CTX",
			question="Q",
			stream=True,
			prefer_converse=True,
		)
		self.assertEqual(out, "STREAM")

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	def test_generate_with_rag_falls_back_when_converse_throws(self, trunc: MagicMock):
		self.brt.converse = MagicMock()

		self.h._generate_with_converse = MagicMock(side_effect=RuntimeError("boom"))
		self.h._generate_with_claude_message_format = MagicMock(return_value="FALLBACK")

		out = self.h.generate_with_rag(
			system_prompt="SYS",
			context="CTX",
			question="Q",
			stream=False,
			prefer_converse=True,
		)
		self.assertEqual(out, "FALLBACK")

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	def test_generate_with_rag_skips_converse_when_prefer_false(self, trunc: MagicMock):
		# even if converse exists, prefer_converse=False should route to fallback
		self.brt.converse = MagicMock()

		self.h._generate_with_claude_message_format = MagicMock(return_value="FALLBACK")
		out = self.h.generate_with_rag(
			system_prompt="SYS",
			context="CTX",
			question="Q",
			prefer_converse=False,
		)
		self.assertEqual(out, "FALLBACK")


# ============================================================
# Converse implementations tests
# ============================================================

class TestBedrockHelperConverseImplementations(TestCase):
	def setUp(self) -> None:
		self.brt = MagicMock()
		self.h = BedrockHelper(
			bedrock_runtime_client=self.brt,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	@patch("bedrockhelper.main.build_converse_request", return_value={"modelId": "m"})
	@patch("bedrockhelper.main.extract_converse_text", return_value="HELLO")
	def test_generate_with_converse_happy_path(self, ex: MagicMock, bcr: MagicMock):
		self.brt.converse = MagicMock(return_value={"output": "x", "ResponseMetadata": {"HTTPHeaders": {}}})

		out = self.h._generate_with_converse(
			model_id="m",
			system_prompt="SYS",
			context="CTX",
			question="Q",
			max_tokens=1,
			temperature=0.1,
			rag_instructions="",
		)

		self.assertEqual(out.text, "HELLO")
		self.assertFalse(out.stream)
		bcr.assert_called_once()

	def test_generate_with_converse_stream_raises_if_missing_stream(self):
		self.brt.converse_stream = MagicMock(return_value={})
		with self.assertRaises(RuntimeError):
			self.h._generate_with_converse_stream(
				model_id="m",
				system_prompt="SYS",
				context="CTX",
				question="Q",
				max_tokens=1,
				temperature=0.1,
				rag_instructions="",
			)

	@patch("bedrockhelper.main.build_converse_request", return_value={"modelId": "m"})
	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	def test_generate_with_converse_stream_closes_stream_and_collects(self, it: MagicMock, bcr: MagicMock):
		stream = FakeStream()

		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("a")
			on_text("b")

		it.side_effect = _iter
		self.brt.converse_stream = MagicMock(
			return_value={
				"stream": stream,
				"ResponseMetadata": {
					"HTTPHeaders": {}
				}
			}
		)

		out = self.h._generate_with_converse_stream(
			model_id="m",
			system_prompt="SYS",
			context="CTX",
			question="Q",
			max_tokens=1,
			temperature=0.1,
			rag_instructions="",
		)

		self.assertEqual(out.text, "ab")
		self.assertTrue(out.stream)
		self.assertTrue(stream.closed)


# ============================================================
# Claude invoke_model fallback tests + _extract_text_from_claude
# ============================================================

class TestBedrockHelperClaudeFallback(TestCase):
	def setUp(self) -> None:
		self.brt = MagicMock()
		self.h = BedrockHelper(
			bedrock_runtime_client=self.brt,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	def test_generate_with_claude_message_format_streaming_happy_path(self, it: MagicMock):
		stream = FakeStream()

		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("x")
			on_text("y")

		it.side_effect = _iter

		self.brt.invoke_model_with_response_stream = MagicMock(
			return_value={"body": stream, "ResponseMetadata": {"HTTPHeaders": {}}})

		out = self.h._generate_with_claude_message_format(
			model_id="m",
			system_prompt="SYS",
			context="CTX",
			question="Q",
			stream=True,
			temperature=0.1,
			max_tokens=5,
			rag_instructions="RAG",
			foo="bar",
		)
		self.assertEqual(out.text, "xy")
		self.assertTrue(out.stream)
		self.assertTrue(stream.closed)

		# ensure extra_params included in request body
		args, kwargs = self.brt.invoke_model_with_response_stream.call_args
		body = json.loads(kwargs["body"])
		self.assertEqual(body["foo"], "bar")

	def test_generate_with_claude_message_format_streaming_missing_body_raises(self):
		self.brt.invoke_model_with_response_stream = MagicMock(return_value={})
		with self.assertRaises(RuntimeError):
			self.h._generate_with_claude_message_format(
				model_id="m",
				system_prompt="SYS",
				context="CTX",
				question="Q",
				stream=True,
				temperature=0.1,
				max_tokens=5,
			)

	def test_generate_with_claude_message_format_nonstream_json_ok(self):
		payload = {
			"content": [
				{"text": "hi"},
				{"text": "!"}
			],
			"usage": {
				"inputTokens": 1,
				"outputTokens": 2
			}
		}
		self.brt.invoke_model = MagicMock(
			return_value={
				"body": FakeBody(json.dumps(payload).encode("utf-8")),
				"ResponseMetadata": {"HTTPHeaders": {}}
			}
		)

		out = self.h._generate_with_claude_message_format(
			model_id="m",
			system_prompt="SYS",
			context="CTX",
			question="Q",
			stream=False,
			temperature=0.1,
			max_tokens=5,
		)
		self.assertEqual(out.text, "hi!")
		self.assertFalse(out.stream)

	def test_generate_with_claude_message_format_nonstream_json_parse_fails(self):
		self.brt.invoke_model = MagicMock(
			return_value={"body": FakeBody(b"not-json"), "ResponseMetadata": {"HTTPHeaders": {}}})

		out = self.h._generate_with_claude_message_format(
			model_id="m",
			system_prompt="SYS",
			context="CTX",
			question="Q",
			stream=False,
			temperature=0.1,
			max_tokens=5,
		)
		# in this case body_json is {"raw": "..."} and extract_text_from_claude returns ""
		self.assertEqual(out.text, "")

	def test_extract_text_from_claude_completion_branch(self):
		self.assertEqual(BedrockHelper._extract_text_from_claude({"completion": "yo"}), "yo")

	def test_extract_text_from_claude_string_branch(self):
		self.assertEqual(BedrockHelper._extract_text_from_claude("hello"), "hello")

	def test_extract_text_from_claude_default_empty(self):
		self.assertEqual(BedrockHelper._extract_text_from_claude({"nope": 1}), "")


# ============================================================
# async_stream_generate_with_rag tests
# ============================================================

class TestBedrockHelperAsyncStream(IsolatedAsyncioTestCase):
	async def asyncSetUp(self) -> None:
		self.brt = MagicMock()
		self.h = BedrockHelper(
			bedrock_runtime_client=self.brt,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
			max_concurrent_streams=2,
		)
		# replace semaphore with counting one to assert acquire/release
		self.h._stream_sema = CountingSemaphore()

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	@patch("bedrockhelper.main.format_context_passages", return_value="FORMATTED")
	@patch("bedrockhelper.main.build_converse_request", return_value={"modelId": "m"})
	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	async def test_prefers_converse_stream_and_yields(
			self,
			it: MagicMock,
			bcr: MagicMock,
			fmt: MagicMock,
			trunc: MagicMock
	):
		stream = FakeStream()

		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("a")
			on_text("b")

		it.side_effect = _iter
		self.brt.converse_stream = MagicMock(return_value={"stream": stream})

		parts: List[str] = []
		async for chunk in self.h.async_stream_generate_with_rag(
				system_prompt="SYS",
				context=["x", "y"],
				include_headers_in_context=True,
				question="Q",
				prefer_converse=True,
				rag_instructions="R",
				foo="bar",
		):
			parts.append(chunk)

		self.assertEqual("".join(parts), "ab")
		self.assertTrue(stream.closed)
		self.assertEqual(self.h._stream_sema.acquire_calls, 1)
		self.assertEqual(self.h._stream_sema.release_calls, 1)
		bcr.assert_called_once()
		# ensure extra param made it through to build_converse_request
		_, kwargs = bcr.call_args
		self.assertEqual(kwargs["foo"], "bar")

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	async def test_falls_back_when_converse_stream_missing_stream(self, it: MagicMock, trunc: MagicMock):
		# converse_stream present but returns missing "stream" -> should fall back to invoke stream
		self.brt.converse_stream = MagicMock(return_value={})

		event_stream = FakeStream()

		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("x")

		it.side_effect = _iter
		self.brt.invoke_model_with_response_stream = MagicMock(return_value={"body": event_stream})

		parts: List[str] = []
		async for chunk in self.h.async_stream_generate_with_rag(
				system_prompt="SYS",
				context="CTX",
				question="Q",
				prefer_converse=True,
		):
			parts.append(chunk)

		self.assertEqual("".join(parts), "x")
		self.assertTrue(event_stream.closed)
		self.assertEqual(self.h._stream_sema.acquire_calls, 1)
		self.assertEqual(self.h._stream_sema.release_calls, 1)

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	async def test_async_stream_raises_when_fallback_missing_body(self, trunc: MagicMock):
		# no converse_stream, fallback invoked but missing body => should raise
		# ensure hasattr(converse_stream) false
		if hasattr(self.brt, "converse_stream"):
			delattr(self.brt, "converse_stream")

		self.brt.invoke_model_with_response_stream = MagicMock(return_value={})

		with self.assertRaises(RuntimeError):
			async for _ in self.h.async_stream_generate_with_rag(
					system_prompt="SYS",
					context="CTX",
					question="Q",
					prefer_converse=False,
			):
				pass

		self.assertEqual(self.h._stream_sema.acquire_calls, 1)
		self.assertEqual(self.h._stream_sema.release_calls, 1)

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	@patch("bedrockhelper.main.iter_bedrock_stream_text", side_effect=RuntimeError("stream parse fail"))
	async def test_async_stream_propagates_worker_exception(self, it: MagicMock, trunc: MagicMock):
		# force fallback path
		if hasattr(self.brt, "converse_stream"):
			delattr(self.brt, "converse_stream")

		event_stream = FakeStream()
		self.brt.invoke_model_with_response_stream = MagicMock(return_value={"body": event_stream})

		with self.assertRaises(RuntimeError):
			async for _ in self.h.async_stream_generate_with_rag(
					system_prompt="SYS",
					context="CTX",
					question="Q",
					prefer_converse=False,
			):
				pass

		self.assertTrue(event_stream.closed)
		self.assertEqual(self.h._stream_sema.acquire_calls, 1)
		self.assertEqual(self.h._stream_sema.release_calls, 1)

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	async def test_async_stream_swallows_semaphore_release_error(self, it: MagicMock, trunc: MagicMock):
		# force fallback path (no converse_stream)
		if hasattr(self.brt, "converse_stream"):
			delattr(self.brt, "converse_stream")

		# semaphore where release() raises -> should be swallowed
		self.h._stream_sema = ReleaseRaisesSemaphore()

		event_stream = FakeStream()

		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("ok")

		it.side_effect = _iter
		self.brt.invoke_model_with_response_stream = MagicMock(return_value={"body": event_stream})

		parts: List[str] = []
		async for chunk in self.h.async_stream_generate_with_rag(
				system_prompt="SYS",
				context="CTX",
				question="Q",
				prefer_converse=False,
		):
			parts.append(chunk)

		self.assertEqual("".join(parts), "ok")
		self.assertTrue(event_stream.closed)
		self.assertEqual(self.h._stream_sema.acquire_calls, 1)
		self.assertEqual(self.h._stream_sema.release_calls, 1)


# ============================================================
# Embeddings tests
# ============================================================

class TestBedrockHelperEmbeddings(TestCase):
	def setUp(self) -> None:
		self.brt = MagicMock()
		self.h = BedrockHelper(
			bedrock_runtime_client=self.brt,
			bedrock_client=MagicMock(),
			s3_client=MagicMock(),
		)

	@patch("bedrockhelper.main.normalize_records", return_value=[])
	def test_embed_texts_empty_returns_empty_response(self, norm: MagicMock):
		out = self.h.embed_texts(records=[])
		self.assertEqual(out.embeddings, {})
		self.assertEqual(out.metrics, {})

	@patch("bedrockhelper.main.normalize_records", return_value=[("a", "A"), ("b", "B")])
	def test_embed_texts_below_threshold_uses_sync_concurrent(self, norm: MagicMock):
		self.h._embed_texts_sync_concurrent = MagicMock(return_value="SYNC")
		out = self.h.embed_texts(records=[("x", "y")], batch_threshold=3000)
		self.assertEqual(out, "SYNC")

	@patch("bedrockhelper.main.normalize_records", return_value=[("a", "A"), ("b", "B")])
	def test_embed_texts_above_threshold_uses_batch_job(self, norm: MagicMock):
		self.h.submit_embedding_batch_job = MagicMock(return_value="BATCH")
		out = self.h.embed_texts(records=[("x", "y")], batch_threshold=1, s3_bucket="b", role_arn="r")
		self.assertEqual(out, "BATCH")

	def test_embed_one_raises_if_embedding_not_list(self):
		self.brt.invoke_model = MagicMock(
			return_value={"body": FakeBody(json.dumps({"embedding": "nope"}).encode("utf-8")),
			              "ResponseMetadata": {"HTTPHeaders": {}}})
		with self.assertRaises(RuntimeError):
			self.h._embed_one("rid", "text")

	def test_embed_one_happy_path(self):
		self.brt.invoke_model = MagicMock(
			return_value={"body": FakeBody(json.dumps({"embedding": [1.0, 2.0]}).encode("utf-8")),
			              "ResponseMetadata": {"HTTPHeaders": {}}})
		rid, emb, metrics = self.h._embed_one("rid", "text")
		self.assertEqual(rid, "rid")
		self.assertEqual(emb, [1.0, 2.0])
		self.assertIsNotNone(metrics)

	def test_embed_texts_sync_concurrent_sequential_branch(self):
		# max_workers <= 1 triggers sequential branch
		self.h._embed_one = MagicMock(side_effect=[
			("a", [1.0], object()),
			("b", [2.0], object()),
		])
		out = self.h._embed_texts_sync_concurrent([("a", "A"), ("b", "B")], max_workers=1)
		self.assertEqual(out.embeddings["a"], [1.0])
		self.assertEqual(out.embeddings["b"], [2.0])
		self.assertEqual(self.h._embed_one.call_count, 2)

	def test_embed_texts_sync_concurrent_threadpool_branch(self):
		# this should still be deterministic enough for unit test purposes
		self.h._embed_one = MagicMock(side_effect=[
			("a", [1.0], object()),
			("b", [2.0], object()),
		])
		out = self.h._embed_texts_sync_concurrent([("a", "A"), ("b", "B")], max_workers=2)
		self.assertEqual(set(out.embeddings.keys()), {"a", "b"})
		self.assertEqual(self.h._embed_one.call_count, 2)


# ============================================================
# Batch job submission + S3 parsing tests
# ============================================================

class TestBedrockHelperBatchEmbeddings(TestCase):
	def setUp(self) -> None:
		self.brt = MagicMock()
		self.br = MagicMock()
		self.s3 = MagicMock()
		self.h = BedrockHelper(
			bedrock_runtime_client=self.brt,
			bedrock_client=self.br,
			s3_client=self.s3,
		)

	def test_submit_embedding_batch_job_requires_bucket_and_role(self):
		with self.assertRaises(ValueError):
			self.h.submit_embedding_batch_job(
				[("a", "A")], s3_bucket=None, s3_prefix="p", role_arn="r"
			)
		with self.assertRaises(ValueError):
			self.h.submit_embedding_batch_job(
				[("a", "A")], s3_bucket="b", s3_prefix="p", role_arn=None
			)

	@patch("bedrockhelper.main.uuid.uuid4")
	def test_submit_embedding_batch_job_job_id_fallbacks(self, uuid4: MagicMock):
		uuid4.return_value = SimpleNamespace(hex="deadbeef")

		# no recognized job id keys -> should use job_name fallback
		self.br.create_model_invocation_job = MagicMock(return_value={})
		self.s3.upload_file = MagicMock()

		out = self.h.submit_embedding_batch_job(
			[("a", "A")],
			s3_bucket="bucket",
			s3_prefix="pref",
			role_arn="arn:role",
		)
		self.assertEqual(out.job_id, out.job_name)
		self.assertIn("deadbeef", out.job_name)

	@patch("bedrockhelper.main.uuid.uuid4")
	@patch("bedrockhelper.main.os.remove", side_effect=OSError("nope"))
	def test_submit_embedding_batch_job_cleanup_swallows_oserror(self, rm: MagicMock, uuid4: MagicMock):
		uuid4.return_value = SimpleNamespace(hex="deadbeef")
		self.br.create_model_invocation_job = MagicMock(return_value={"jobArn": "arn:job"})
		self.s3.upload_file = MagicMock()

		# should not raise even though os.remove fails
		out = self.h.submit_embedding_batch_job(
			[("a", "A")],
			s3_bucket="bucket",
			s3_prefix="pref",
			role_arn="arn:role",
		)
		self.assertEqual(out.job_id, "arn:job")

	@patch("bedrockhelper.main.normalize_records", return_value=[("a", "A")])
	def test_submit_embedding_batch_job_normalizes_non_list_pairs(self, norm: MagicMock):
		self.br.create_model_invocation_job = MagicMock(return_value={"jobId": "jid"})
		self.s3.upload_file = MagicMock()

		# pass something not a list of tuples to trigger normalize_records branch
		out = self.h.submit_embedding_batch_job(
			pairs={"a": "A"},  # type: ignore[arg-type]
			s3_bucket="bucket",
			s3_prefix="pref",
			role_arn="arn:role",
		)
		self.assertEqual(out.job_id, "jid")
		norm.assert_called_once()

	def test_download_batch_results_jsonl_rejects_non_s3_uri(self):
		with self.assertRaises(ValueError):
			list(self.h.download_batch_results_jsonl(output_s3_uri="http://nope"))

	def test_download_batch_results_jsonl_parses_only_jsonl_and_skips_blanks(self):
		# s3://bucket/prefix (without trailing slash) should be normalized to prefix/
		pages = [
			{"Contents": [
				{"Key": "prefix/ignore.txt"},
				{"Key": "prefix/a.jsonl"},
			]}
		]
		self.s3.get_paginator = MagicMock(return_value=FakePaginator(pages))

		payload = (
			b'\n{"recordId":"1","modelOutput":{"embedding":[1]}}'
			b'\n\n{"recordId":"2","modelOutput":{"embedding":[2]}}\n'
		)
		self.s3.get_object = MagicMock(return_value={"Body": FakeBody(payload)})

		got = list(self.h.download_batch_results_jsonl(output_s3_uri="s3://bucket/prefix"))
		self.assertEqual(len(got), 2)
		self.assertEqual(got[0]["recordId"], "1")
		self.assertEqual(got[1]["recordId"], "2")

	def test_parse_batch_embeddings_covers_all_skips(self):
		# drive parse via a controlled generator
		def fake_iter(**kwargs: Any) -> Iterator[dict]:
			yield {"no_record": True}  # skip (no recordId)
			yield {"recordId": "1", "modelOutput": "not-a-dict"}  # skip (model_out not dict)
			yield {"recordId": "2", "modelOutput": {"embedding": "nope"}}  # skip (not list)
			yield {"record_id": "3", "model_output": {"embedding": [3.0]}}  # accept
			yield {"recordId": "4", "output": {"embedding": [4.0]}}  # accept via 'output'

		self.h.download_batch_results_jsonl = MagicMock(side_effect=lambda **kw: fake_iter(**kw))

		out = self.h.parse_batch_embeddings(output_s3_uri="s3://bucket/prefix/")
		self.assertEqual(out, {"3": [3.0], "4": [4.0]})

	def test_get_batch_job_delegates(self):
		self.br.get_model_invocation_job = MagicMock(return_value={"status": "Completed"})
		out = self.h.get_batch_job("jid")
		self.assertEqual(out["status"], "Completed")
		self.br.get_model_invocation_job.assert_called_once_with(jobIdentifier="jid")

	@patch("bedrockhelper.main.time.sleep", return_value=None)
	def test_wait_for_batch_job_returns_on_terminal_status(self, _sleep: MagicMock):
		# first call InProgress then Completed
		self.h.get_batch_job = MagicMock(side_effect=[
			{"status": "InProgress"},
			{"status": "Completed"},
		])
		out = self.h.wait_for_batch_job("jid", poll_seconds=0.0, timeout_seconds=1.0)
		self.assertEqual(out["status"], "Completed")

	@patch("bedrockhelper.main.time.sleep", return_value=None)
	@patch("bedrockhelper.main.time.time")
	def test_wait_for_batch_job_times_out(self, ttime: MagicMock, _sleep: MagicMock):
		# time advances past deadline quickly
		start = 1000.0
		ttime.side_effect = [start, start + 2.0]  # initial + then beyond deadline
		self.h.get_batch_job = MagicMock(return_value={"status": "InProgress"})

		with self.assertRaises(TimeoutError):
			self.h.wait_for_batch_job("jid", poll_seconds=0.0, timeout_seconds=1.0)

	def test_input_text_token_count_non_int_is_ignored(self):
		resp = {"ResponseMetadata": {"HTTPHeaders": {}}}
		# Non-int value that will raise TypeError when passed to int()
		body = {"inputTextTokenCount": {"not": "an-int"}}

		# Should not raise and input_tokens should remain None
		metrics = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertIsNone(metrics.input_tokens)

	def test_input_text_token_count_string_parsed(self):
		resp = {"ResponseMetadata": {"HTTPHeaders": {}}}
		# Valid numeric string should be parsed to int
		body = {"inputTextTokenCount": "99"}

		metrics = BedrockHelper._extract_metrics_from_response(resp, body)
		self.assertEqual(metrics.input_tokens, 99)

	@patch("bedrockhelper.main.normalize_headers")
	def test_extract_metrics_header_values_none_are_ignored(self, norm_headers: MagicMock):
		norm_headers.return_value = {
			"x-amzn-bedrock-input-token-count": None,
			"x-amzn-bedrock-output-token-count": None,
			"x-amzn-bedrock-invocation-latency": None,
			"x-amzn-bedrock-first-byte-latency": None,
		}

		resp = {"ResponseMetadata": {"HTTPHeaders": {"some-header": "present"}}}
		metrics = BedrockHelper._extract_metrics_from_response(resp, None)

		# Values coming from headers that are None should leave metrics as None
		self.assertIsNone(metrics.input_tokens)
		self.assertIsNone(metrics.output_tokens)
		self.assertIsNone(metrics.invocation_latency_ms)
		self.assertIsNone(metrics.first_byte_latency_ms)

		# header_metrics should reflect the normalized headers mapping
		self.assertEqual(metrics.header_metrics, norm_headers.return_value)


	@patch("bedrockhelper.main.normalize_headers")
	def test_extract_metrics_header_get_int_branch(self, norm_headers: MagicMock):
		# mix of valid int-string, invalid string, int, and a type that will raise in int()
		norm_headers.return_value = {
			"x-amzn-bedrock-input-token-count": "12",
			"x-amzn-bedrock-output-token-count": "not-an-int",
			"x-amzn-bedrock-invocation-latency": 7,
			"x-amzn-bedrock-first-byte-latency": {"bad": "value"},
		}

		resp = {"ResponseMetadata": {"HTTPHeaders": {"irrelevant": "present"}}}
		metrics = BedrockHelper._extract_metrics_from_response(resp, None)

		self.assertEqual(metrics.input_tokens, 12)
		self.assertIsNone(metrics.output_tokens)
		self.assertEqual(metrics.invocation_latency_ms, 7)
		self.assertIsNone(metrics.first_byte_latency_ms)
		# ensure header_metrics preserved
		self.assertEqual(metrics.header_metrics, norm_headers.return_value)


	def test_wait_for_batch_job_propagates_get_batch_job_exception(self):
		# Simulate get_batch_job raising an error; wait_for_batch_job should propagate it immediately.
		self.h.get_batch_job = MagicMock(side_effect=RuntimeError("boom"))
		with self.assertRaises(RuntimeError):
			self.h.wait_for_batch_job("jid", poll_seconds=0.0, timeout_seconds=1.0)


	# python
	def test_wait_for_batch_job_returns_immediately_for_terminal_status(self):
		# Immediate terminal status should return and not call time.sleep
		self.h.get_batch_job = MagicMock(return_value={"status": "Completed"})
		with patch("bedrockhelper.main.time.sleep") as sleep:
			out = self.h.wait_for_batch_job("jid", poll_seconds=0.0, timeout_seconds=1.0)
			self.assertEqual(out["status"], "Completed")
			self.assertEqual(sleep.call_count, 0)


	def test_wait_for_batch_job_handles_various_terminal_statuses_and_polling(self):
		# For each terminal status, simulate one non-terminal poll then terminal,
		# ensure time.sleep is called once. Also test a multi-poll case.
		terminal_statuses = ("Completed", "Failed", "Stopped", "Expired")
		for status in terminal_statuses:
			with self.subTest(status=status):
				self.h.get_batch_job = MagicMock(side_effect=[
					{"status": "InProgress"},
					{"status": status},
				])
				with patch("bedrockhelper.main.time.sleep") as sleep:
					out = self.h.wait_for_batch_job("jid", poll_seconds=0.0, timeout_seconds=1.0)
					self.assertEqual(out["status"], status)
					self.assertEqual(sleep.call_count, 1)

		# multi-poll scenario: two non-terminal polls then terminal -> two sleeps
		self.h.get_batch_job = MagicMock(side_effect=[
			{"status": "InProgress"},
			{"status": "InProgress"},
			{"status": "Completed"},
		])
		with patch("bedrockhelper.main.time.sleep") as sleep:
			out = self.h.wait_for_batch_job("jid", poll_seconds=0.0, timeout_seconds=5.0)
			self.assertEqual(out["status"], "Completed")
			self.assertEqual(sleep.call_count, 2)


	@patch("bedrockhelper.main.uuid.uuid4")
	@patch("bedrockhelper.main.os.remove")
	def test_submit_embedding_batch_job_uses_jobIdentifier_and_removes_tempfile(self, rm: MagicMock, uuid4: MagicMock):
		uuid4.return_value = SimpleNamespace(hex="deadbeef")
		self.br.create_model_invocation_job = MagicMock(return_value={"jobIdentifier": "jid-123"})
		self.s3.upload_file = MagicMock()

		out = self.h.submit_embedding_batch_job(
			[("a", "A")],
			s3_bucket="bucket",
			s3_prefix="pref",
			role_arn="arn:role",
		)

		self.assertEqual(out.job_id, "jid-123")

		# ensure the temp file cleanup "happy path" executed (covers the os.remove line)
		rm.assert_called_once()
		args, _ = rm.call_args
		self.assertTrue(isinstance(args[0], str))
		self.assertTrue(args[0].endswith(".jsonl"))


	@patch("bedrockhelper.main.uuid.uuid4")
	def test_submit_embedding_batch_job_uses_id_key(self, uuid4: MagicMock):
		uuid4.return_value = SimpleNamespace(hex="deadbeef")
		self.br.create_model_invocation_job = MagicMock(return_value={"id": "id-999"})
		self.s3.upload_file = MagicMock()

		out = self.h.submit_embedding_batch_job(
			[("a", "A")],
			s3_bucket="bucket",
			s3_prefix="pref",
			role_arn="arn:role",
		)

		self.assertEqual(out.job_id, "id-999")


class TestAsyncGenerateWithRag(IsolatedAsyncioTestCase):
	async def asyncSetUp(self) -> None:
		self.brt = MagicMock()
		self.br = MagicMock()
		self.s3 = MagicMock()
		self.h = BedrockHelper(
			bedrock_runtime_client=self.brt,
			bedrock_client=self.br,
			s3_client=self.s3,
		)

	async def test_async_generate_with_rag_delegates_to_generate_with_rag(self):
		sentinel = RAGResponse(text="ok", stream=False, metrics=None, raw_response=None)

		with patch.object(BedrockHelper, "generate_with_rag", return_value=sentinel) as gen:
			out = await self.h.async_generate_with_rag(
				system_prompt="SYS",
				context="CTX",
				question="Q",
				stream=False,
				foo="bar",
			)

			# ensure the async wrapper returned the same object
			self.assertIs(out, sentinel)

			# ensure the sync method was called once with kwargs forwarded
			gen.assert_called_once()
			_, kwargs = gen.call_args
			self.assertEqual(kwargs["system_prompt"], "SYS")
			self.assertEqual(kwargs["context"], "CTX")
			self.assertEqual(kwargs["question"], "Q")
			self.assertEqual(kwargs["stream"], False)
			self.assertEqual(kwargs["foo"], "bar")

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	async def test_async_stream_no_rag_instructions_prefix(self, it: MagicMock, trunc: MagicMock):
		if hasattr(self.brt, "converse_stream"):
			delattr(self.brt, "converse_stream")

		event_stream = FakeStream()

		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("x")

		it.side_effect = _iter
		self.brt.invoke_model_with_response_stream = MagicMock(return_value={"body": event_stream})

		parts: List[str] = []
		async for chunk in self.h.async_stream_generate_with_rag(
				system_prompt="SYS",
				context="CTX",
				question="Q",
				prefer_converse=False,
				rag_instructions="   ",  # blank after strip()
		):
			parts.append(chunk)

		self.assertEqual("".join(parts), "x")

		# verify request did NOT get a "prefix\n" before Context:
		_, kwargs = self.brt.invoke_model_with_response_stream.call_args
		body = json.loads(kwargs["body"])
		text = body["messages"][0]["content"][0]["text"]
		self.assertTrue(text.startswith("Context:\nCTX\n\nQuestion:\nQ\n"))

	@patch("bedrockhelper.main.truncate_by_chars", side_effect=lambda s, n: s)
	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	async def test_async_stream_includes_trimmed_rag_instructions_prefix(self, it: MagicMock, trunc: MagicMock):
		# force fallback path (no converse_stream)
		if hasattr(self.brt, "converse_stream"):
			delattr(self.brt, "converse_stream")

		# replace the helper's semaphore with the counting test double so we can assert calls
		self.h._stream_sema = CountingSemaphore()

		event_stream = FakeStream()

		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("x")

		it.side_effect = _iter
		self.brt.invoke_model_with_response_stream = MagicMock(return_value={"body": event_stream})

		parts: List[str] = []
		# rag_instructions has surrounding whitespace to verify .strip() is applied
		rag_instructions = "  RAG-INSTR  "

		async for chunk in self.h.async_stream_generate_with_rag(
				system_prompt="SYS",
				context="CTX",
				question="Q",
				prefer_converse=False,
				rag_instructions=rag_instructions,
		):
			parts.append(chunk)

		# basic stream behavior checks
		self.assertEqual("".join(parts), "x")
		self.assertTrue(event_stream.closed)
		self.assertEqual(self.h._stream_sema.acquire_calls, 1)
		self.assertEqual(self.h._stream_sema.release_calls, 1)

		# ensure the body included the trimmed rag_instructions as a prefix + newline
		self.brt.invoke_model_with_response_stream.assert_called_once()
		_, kwargs = self.brt.invoke_model_with_response_stream.call_args
		body = json.loads(kwargs["body"])
		text = body["messages"][0]["content"][0]["text"]
		# expected prefix is the stripped instructions followed by a newline, then "Context:"
		self.assertTrue(text.startswith("RAG-INSTR\nContext:\nCTX\n\nQuestion:\nQ\n"))

	@patch("bedrockhelper.main.iter_bedrock_stream_text")
	async def test_async_stream_exta_params_forwarded(self, it: MagicMock):
		# force fallback path (no converse_stream)
		if hasattr(self.brt, "converse_stream"):
			delattr(self.brt, "converse_stream")

		event_stream = FakeStream()

		# make the patched iterator call on_text once so the async generator yields
		def _iter(stream_obj: Any, on_text: Any, stream_kind: str) -> None:
			on_text("x")

		it.side_effect = _iter
		self.brt.invoke_model_with_response_stream = MagicMock(return_value={"body": event_stream})

		parts: List[str] = []
		async for chunk in self.h.async_stream_generate_with_rag(
				system_prompt="SYS",
				context="CTX",
				question="Q",
				prefer_converse=False,
				foo="bar",
		):
			parts.append(chunk)

		# ensure extra param made it through to invoke_model_with_response_stream
		self.brt.invoke_model_with_response_stream.assert_called_once()
		_, kwargs = self.brt.invoke_model_with_response_stream.call_args
		body = json.loads(kwargs["body"])
		self.assertEqual(body["foo"], "bar")

	async def test_async_embed_texts_delegates_to_embed_texts(self):
		sentinel = object()
		with patch.object(BedrockHelper, "embed_texts", return_value=sentinel) as em:
			out = await self.h.async_embed_texts(
				records=[("rid", "text")],
				batch_threshold=1,
				max_workers=2,
			)

			# ensure the async wrapper returned the same object
			self.assertIs(out, sentinel)

			# ensure the sync method was called once with kwargs forwarded
			em.assert_called_once()
			_, kwargs = em.call_args
			self.assertEqual(kwargs["records"], [("rid", "text")])
			self.assertEqual(kwargs["batch_threshold"], 1)
			self.assertEqual(kwargs["max_workers"], 2)

