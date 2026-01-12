"""
A helper module for interacting with Amazon Bedrock for retrieval-augmented
generation (RAG) and for producing vector embeddings.

Design goals:
- Works in both sync + async contexts.
- Defaults to Bedrock Runtime Converse/ConverseStream for chat-style RAG Q&A
  (when available), with a robust fallback to invoke_model /
  invoke_model_with_response_stream.
- Embeddings via Titan embed text v2:
  - small batches: synchronous invoke_model (optionally concurrent)
  - large batches: S3 JSONL + create_model_invocation_job (batch inference)
- IAM-role based auth via boto3 default credential resolution (no raw keys).

Note:
- Converse APIs are only for message/chat interactions. Embeddings are handled
  via invoke_model/batch inference.
- Batch inference requires account enablement and correct role/bucket perms.
"""

from __future__ import annotations
import time
import asyncio
import json
import logging
import os
import tempfile
import threading
import uuid
from bedrockhelper.models import (
	InvocationMetrics,
	RAGResponse,
	EmbeddingResponse,
	BatchJobResponse,
)

from typing import (
	Any,
	AsyncIterator,
	Dict,
	Iterator,
	List,
	Optional,
	Sequence,
	Tuple,
	Union,
)

import boto3
from botocore.config import Config

from bedrockhelper.utils import normalize_records, truncate_by_chars, format_context_passages, normalize_headers, \
	iter_bedrock_stream_text, extract_converse_text, build_converse_request
from .types import RecordInput
log = logging.getLogger(__name__)


class BedrockHelper:
	"""
	Utility class for interacting with Amazon Bedrock.

	Core methods are synchronous (boto3 is sync).
	Async wrappers are provided for event-loop safe usage.

	Parameters
	----------
	region_name:
		AWS region for Bedrock.
	rag_model_id:
		Default model id for RAG chat (Claude, etc.).
	embedding_model_id:
		Default model id for embeddings (Titan v2).
	botocore_config:
		Optional botocore Config for retries/timeouts.
	s3_client, bedrock_runtime_client, bedrock_client:
		Optional preconfigured clients (for testing/custom endpoints).
	"""

	def __init__(
			self,
			region_name: str = 'ca-central-1',
			rag_model_id: str = 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
			embedding_model_id: str = 'amazon.titan-embed-text-v2:0',
			*,
			botocore_config: Optional[Config] = None,
			s3_client: Optional[Any] = None,
			bedrock_runtime_client: Optional[Any] = None,
			bedrock_client: Optional[Any] = None,
			max_concurrent_streams: int = 8,
	) -> None:
		if max_concurrent_streams < 1:
			raise ValueError("max_concurrent_streams must be >= 1")

		self._max_concurrent_streams = max_concurrent_streams
		self._stream_sema = threading.BoundedSemaphore(max_concurrent_streams)

		if botocore_config is None:
			retries: Any = {'max_attempts': 10, 'mode': 'adaptive'}
			botocore_config = Config(
				retries=retries,
				connect_timeout=5,
				read_timeout=120,
			)

		self.bedrock_runtime = bedrock_runtime_client or boto3.client(
			'bedrock-runtime',
			region_name=region_name,
			config=botocore_config,
		)
		self.bedrock = bedrock_client or boto3.client(
			'bedrock',
			region_name=region_name,
			config=botocore_config,
		)
		self.s3 = s3_client or boto3.client(
			's3',
			region_name=region_name,
			config=botocore_config,
		)

		self.rag_model_id = rag_model_id
		self.embedding_model_id = embedding_model_id

	# ------------------------------------------------------------------
	# Metrics helpers
	# ------------------------------------------------------------------

	@classmethod
	def _extract_metrics_from_response(
			cls,
			response: dict,
			response_body: Optional[dict] = None,
	) -> InvocationMetrics:
		metrics = InvocationMetrics()

		# ---- Body metrics (model-specific)
		if response_body and isinstance(response_body, dict):
			usage = response_body.get('usage')
			if isinstance(usage, dict):
				metrics.usage = usage

				def _pick_int(*keys: str) -> Optional[int]:
					for k in keys:
						v = usage.get(k)
						if v is None:
							continue
						try:
							return int(v)
						except (TypeError, ValueError):
							pass
					return None

				metrics.input_tokens = _pick_int(
					'inputTokens',
					'prompt_tokens',
					'promptTokens',
					'input_tokens',
				)
				metrics.output_tokens = _pick_int(
					'outputTokens',
					'completion_tokens',
					'completionTokens',
					'output_tokens',
				)
				metrics.total_tokens = _pick_int('totalTokens', 'total_tokens')

			# Titan embed frequently uses inputTextTokenCount rather than a usage dict
			if metrics.input_tokens is None:
				itc = response_body.get('inputTextTokenCount')
				if itc is not None:
					try:
						metrics.input_tokens = int(itc)
					except (TypeError, ValueError):
						pass

			# Some responses embed invocation metrics under this key
			inv = response_body.get('amazon-bedrock-invocationMetrics')
			if isinstance(inv, dict):
				metrics.invocation_latency_ms = inv.get('invocationLatency') or metrics.invocation_latency_ms
				metrics.first_byte_latency_ms = inv.get('firstByteLatency') or metrics.first_byte_latency_ms

		# ---- Header metrics
		http_headers = response.get('ResponseMetadata', {}).get('HTTPHeaders', {}) or {}
		if http_headers:
			norm = normalize_headers(http_headers)
			metrics.header_metrics = dict(norm)

			def _get_int(name: str) -> Optional[int]:
				v = norm.get(name.lower())
				if v is None:
					return None
				try:
					return int(v)
				except (TypeError, ValueError):
					return None

			if metrics.input_tokens is None:
				metrics.input_tokens = _get_int('x-amzn-bedrock-input-token-count')
			if metrics.output_tokens is None:
				metrics.output_tokens = _get_int('x-amzn-bedrock-output-token-count')
			if metrics.invocation_latency_ms is None:
				metrics.invocation_latency_ms = _get_int('x-amzn-bedrock-invocation-latency')
			if metrics.first_byte_latency_ms is None:
				metrics.first_byte_latency_ms = _get_int('x-amzn-bedrock-first-byte-latency')

		if metrics.total_tokens is None and metrics.input_tokens is not None and metrics.output_tokens is not None:
			metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

		return metrics

	# ------------------------------------------------------------------
	# Public RAG API
	# ------------------------------------------------------------------

	def generate_with_rag(
			self,
			*,
			system_prompt: str,
			context: Union[str, Sequence[str]],
			include_headers_in_context: bool = False,
			question: str,
			model_id: Optional[str] = None,
			stream: bool = False,
			temperature: float = 0.1,
			max_tokens: int = 1024,
			max_context_chars: Optional[int] = 120_000,
			prefer_converse: bool = True,
			rag_instructions: str = '',
			**extra_params: Any,
	) -> RAGResponse:
		"""
		Run a RAG-style Q&A call.

		Default behavior:
		- If prefer_converse=True and the boto3 client exposes converse/converse_stream,
		  we use it first.
		- If it errors (or isn't present), we fall back to Claude message format via
		  invoke_model/invoke_model_with_response_stream.

		Returns:
			RAGResponse(text, stream, metrics, raw_response)
		"""
		selected_model = model_id or self.rag_model_id

		if isinstance(context, str):
			ctx = context
		else:
			ctx = format_context_passages(list(context), include_headers=include_headers_in_context)

		ctx = truncate_by_chars(ctx, max_context_chars)

		# 1) Prefer Converse if present
		if prefer_converse and hasattr(self.bedrock_runtime, 'converse'):
			try:
				if stream and hasattr(self.bedrock_runtime, 'converse_stream'):
					return self._generate_with_converse_stream(
						model_id=selected_model,
						system_prompt=system_prompt,
						context=ctx,
						question=question,
						max_tokens=max_tokens,
						temperature=temperature,
						rag_instructions=rag_instructions,
						**extra_params,
					)
				else:
					return self._generate_with_converse(
						model_id=selected_model,
						system_prompt=system_prompt,
						context=ctx,
						question=question,
						max_tokens=max_tokens,
						temperature=temperature,
						rag_instructions=rag_instructions,
						**extra_params,
					)
			except Exception:
				# Converse sometimes fails due to model incompat, param mismatch, etc.
				# We fallback to invoke_model path.
				log.debug('Converse path failed; falling back to invoke_model.', exc_info=True)

		# 2) Fallback: Claude message format
		return self._generate_with_claude_message_format(
			model_id=selected_model,
			system_prompt=system_prompt,
			context=ctx,
			question=question,
			stream=stream,
			temperature=temperature,
			max_tokens=max_tokens,
			rag_instructions=rag_instructions,
			**extra_params,
		)

	# ------------------------------------------------------------------
	# Converse implementations
	# ------------------------------------------------------------------

	def _generate_with_converse(
			self,
			*,
			model_id: str,
			system_prompt: str,
			context: str,
			question: str,
			max_tokens: int,
			temperature: float,
			rag_instructions: str,
			**extra_params: Any,
	) -> RAGResponse:
		req = build_converse_request(
			model_id=model_id,
			system_prompt=system_prompt,
			context=context,
			question=question,
			max_tokens=max_tokens,
			temperature=temperature,
			rag_instructions=rag_instructions,
			**extra_params,
		)

		resp = self.bedrock_runtime.converse(**req)
		text = extract_converse_text(resp if isinstance(resp, dict) else {})
		metrics = self._extract_metrics_from_response(resp, resp if isinstance(resp, dict) else None)
		return RAGResponse(text=text, stream=False, metrics=metrics, raw_response=resp)

	def _generate_with_converse_stream(
			self,
			*,
			model_id: str,
			system_prompt: str,
			context: str,
			question: str,
			max_tokens: int,
			temperature: float,
			rag_instructions: str,
			**extra_params: Any,
	) -> RAGResponse:
		req = build_converse_request(
			model_id=model_id,
			system_prompt=system_prompt,
			context=context,
			question=question,
			max_tokens=max_tokens,
			temperature=temperature,
			rag_instructions=rag_instructions,
			**extra_params,
		)

		resp = self.bedrock_runtime.converse_stream(**req)
		stream_obj = resp.get("stream")
		if stream_obj is None:
			raise RuntimeError("converse_stream response missing 'stream'")

		chunks: List[str] = []
		try:
			# Reuse your unified parser:
			iter_bedrock_stream_text(stream_obj, on_text=chunks.append, stream_kind="converse")
		finally:
			if hasattr(stream_obj, "close"):
				stream_obj.close()

		text = "".join(chunks)
		metrics = self._extract_metrics_from_response(resp, None)
		return RAGResponse(text=text, stream=True, metrics=metrics, raw_response=None)

	# ------------------------------------------------------------------
	# invoke_model fallback (Claude message format)
	# ------------------------------------------------------------------

	def _generate_with_claude_message_format(
			self,
			*,
			model_id: str,
			system_prompt: str,
			context: str,
			question: str,
			stream: bool,
			temperature: float,
			max_tokens: int,
			rag_instructions: str = '',
			**extra_params: Any,
	) -> RAGResponse:
		user_text = f"{rag_instructions}\nContext:\n{context}\n\nQuestion:\n{question}\n"

		messages = [
			{
				"role": "user",
				"content": [{"type": "text", "text": user_text}],
			}
		]

		request_body: Dict[str, Any] = {
			"anthropic_version": "bedrock-2023-05-31",
			"system": system_prompt,
			"messages": messages,
			"max_tokens": max_tokens,
			"temperature": temperature,
		}
		if extra_params:
			request_body.update(extra_params)

		if stream:
			resp = self.bedrock_runtime.invoke_model_with_response_stream(
				modelId=model_id,
				body=json.dumps(request_body),
			)
			event_stream = resp.get("body")
			if event_stream is None:
				raise RuntimeError("invoke_model_with_response_stream response missing 'body'")

			chunks: List[str] = []
			try:
				iter_bedrock_stream_text(event_stream, on_text=chunks.append, stream_kind="invoke")
			finally:
				if hasattr(event_stream, "close"):
					event_stream.close()

			text = "".join(chunks)
			metrics = self._extract_metrics_from_response(resp, None)
			return RAGResponse(text=text, stream=True, metrics=metrics, raw_response=None)

		resp = self.bedrock_runtime.invoke_model(
			modelId=model_id,
			body=json.dumps(request_body),
		)
		body_bytes = resp.get("body").read()
		try:
			body_json = json.loads(body_bytes)
		except Exception:
			body_json = {"raw": body_bytes.decode("utf-8", errors="replace")}

		text = self._extract_text_from_claude(body_json)
		metrics = self._extract_metrics_from_response(resp, body_json if isinstance(body_json, dict) else None)
		return RAGResponse(text=text, stream=False, metrics=metrics, raw_response=body_json)

	@staticmethod
	def _extract_text_from_claude(body: Any) -> str:
		if isinstance(body, dict):
			content = body.get('content')
			if isinstance(content, list):
				parts: List[str] = []
				for item in content:
					if isinstance(item, dict):
						t = item.get('text')
						if isinstance(t, str):
							parts.append(t)
				return ''.join(parts)

			comp = body.get('completion')
			if isinstance(comp, str):
				return comp

		if isinstance(body, str):
			return body
		return ''

	# ------------------------------------------------------------------
	# Async wrappers for RAG
	# ------------------------------------------------------------------

	async def async_generate_with_rag(self, **kwargs: Any) -> RAGResponse:
		return await asyncio.to_thread(self.generate_with_rag, **kwargs)

	async def async_stream_generate_with_rag(self, **kwargs: Any) -> AsyncIterator[str]:
		"""
		Async generator that streams Bedrock text deltas.

		- Prefers ConverseStream when available (and prefer_converse=True).
		- Falls back to invoke_model_with_response_stream using Claude message format.
		- Raises on error (does NOT yield error text).
		- Limits concurrent streams to avoid unbounded thread creation.
		"""
		kwargs = dict(kwargs)
		kwargs["stream"] = True

		loop = asyncio.get_running_loop()
		q: asyncio.Queue[Optional[str]] = asyncio.Queue()

		# Store exception from worker thread (if any). We'll raise after draining/terminating.
		err: Dict[str, BaseException] = {}

		def push_text(t: str) -> None:
			loop.call_soon_threadsafe(q.put_nowait, t)

		def push_done() -> None:
			loop.call_soon_threadsafe(q.put_nowait, None)

		def push_error(e: BaseException) -> None:
			err["exc"] = e
			loop.call_soon_threadsafe(q.put_nowait, None)

		RESERVED_KEYS = {
			"system_prompt",
			"context",
			"include_headers_in_context",
			"question",
			"model_id",
			"stream",
			"temperature",
			"max_tokens",
			"max_context_chars",
			"prefer_converse",
			"rag_instructions",
		}

		def extra_params_from_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
			return {k: v for k, v in d.items() if k not in RESERVED_KEYS}

		def normalize_context_value(
				context_value: Union[str, Sequence[str]],
				*,
				include_headers_in_context: bool,
				max_context_chars: Optional[int],
		) -> str:
			if isinstance(context_value, str):
				ctx = context_value
			else:
				ctx = format_context_passages(
					list(context_value),
					include_headers=include_headers_in_context,
				)
			return truncate_by_chars(ctx, max_context_chars)

		def _worker() -> None:
			acquired = False
			try:
				# ---- concurrency guard
				self._stream_sema.acquire()
				acquired = True

				selected_model = kwargs.get("model_id") or self.rag_model_id
				system_prompt = kwargs["system_prompt"]
				context_value = kwargs["context"]
				include_headers_in_context = kwargs.get("include_headers_in_context", False)
				question = kwargs["question"]

				# Match sync defaults
				temperature = kwargs.get("temperature", 0.1)
				max_tokens = kwargs.get("max_tokens", 1024)
				max_context_chars = kwargs.get("max_context_chars", 120_000)
				prefer_converse = kwargs.get("prefer_converse", True)
				rag_instructions = kwargs.get("rag_instructions", "")

				ctx = normalize_context_value(
					context_value,
					include_headers_in_context=include_headers_in_context,
					max_context_chars=max_context_chars,
				)

				extra_params = extra_params_from_kwargs(kwargs)

				# 1) Prefer converse_stream (same builder as sync)
				if prefer_converse and hasattr(self.bedrock_runtime, "converse_stream"):
					try:
						req = build_converse_request(
							model_id=selected_model,
							system_prompt=system_prompt,
							context=ctx,
							question=question,
							max_tokens=max_tokens,
							temperature=temperature,
							rag_instructions=rag_instructions,
							**extra_params,
						)

						resp = self.bedrock_runtime.converse_stream(**req)
						stream_obj = resp.get("stream")
						if stream_obj is None:
							raise RuntimeError("converse_stream response missing 'stream'")

						try:
							iter_bedrock_stream_text(stream_obj, on_text=push_text, stream_kind="converse")
						finally:
							if hasattr(stream_obj, "close"):
								stream_obj.close()

						push_done()
						return
					except Exception:
						log.debug("converse_stream failed; falling back.", exc_info=True)

				# 2) Fallback: invoke_model_with_response_stream (Claude message format)
				# Mirror _generate_with_claude_message_format's user_text behavior
				prefix = ""
				if isinstance(rag_instructions, str) and rag_instructions.strip():
					prefix = rag_instructions.strip() + "\n"
				user_text = f"{prefix}Context:\n{ctx}\n\nQuestion:\n{question}\n"

				body: Dict[str, Any] = {
					"anthropic_version": "bedrock-2023-05-31",
					"system": system_prompt,
					"messages": [
						{
							"role": "user",
							"content": [{"type": "text", "text": user_text}],
						}
					],
					"max_tokens": max_tokens,
					"temperature": temperature,
				}
				if extra_params:
					body.update(extra_params)

				resp = self.bedrock_runtime.invoke_model_with_response_stream(
					modelId=selected_model,
					body=json.dumps(body),
				)
				event_stream = resp.get("body")
				if event_stream is None:
					raise RuntimeError("invoke_model_with_response_stream response missing 'body'")

				try:
					iter_bedrock_stream_text(event_stream, on_text=push_text, stream_kind="invoke")
				finally:
					if hasattr(event_stream, "close"):
						event_stream.close()

				push_done()

			except BaseException as e:
				# propagate the original exception to the async generator
				push_error(e)
			finally:
				if acquired:
					try:
						self._stream_sema.release()
					except Exception:
						# Should never happen with BoundedSemaphore unless double-release;
						# swallow to avoid masking original errors.
						pass

		threading.Thread(target=_worker, daemon=True).start()

		while True:
			item = await q.get()
			if item is None:
				break
			yield item

		if "exc" in err:
			raise err["exc"]

	# ------------------------------------------------------------------
	# Embeddings
	# ------------------------------------------------------------------

	def _embed_one(self, record_id: str, text: str) -> Tuple[str, List[float], InvocationMetrics]:
		req = {'inputText': text}
		resp = self.bedrock_runtime.invoke_model(
			modelId=self.embedding_model_id,
			body=json.dumps(req),
		)
		body_bytes = resp.get('body').read()
		data = json.loads(body_bytes)

		embedding = data.get('embedding')
		if not isinstance(embedding, list):
			raise RuntimeError(f'Unexpected embedding response for record {record_id}: {data}')

		metrics = self._extract_metrics_from_response(resp, data)
		return record_id, embedding, metrics

	def embed_texts(
			self,
			records: RecordInput,
			*,
			batch_threshold: int = 3000,
			s3_bucket: Optional[str] = None,
			s3_prefix: str = 'bedrock_batch',
			role_arn: Optional[str] = None,
			max_workers: int = 8,
	) -> Union[EmbeddingResponse, BatchJobResponse]:
		pairs = normalize_records(records)
		if not pairs:
			return EmbeddingResponse(embeddings={}, metrics={})

		if len(pairs) <= batch_threshold:
			return self._embed_texts_sync_concurrent(pairs, max_workers=max_workers)

		return self.submit_embedding_batch_job(
			pairs,
			s3_bucket=s3_bucket,
			s3_prefix=s3_prefix,
			role_arn=role_arn,
		)

	def _embed_texts_sync_concurrent(
			self,
			pairs: List[Tuple[str, str]],
			*,
			max_workers: int,
	) -> EmbeddingResponse:
		from concurrent.futures import ThreadPoolExecutor, as_completed

		embeddings: Dict[str, List[float]] = {}
		metrics_map: Dict[str, InvocationMetrics] = {}

		if max_workers <= 1 or len(pairs) == 1:
			for rid, text in pairs:
				rid2, emb, met = self._embed_one(rid, text)
				embeddings[rid2] = emb
				metrics_map[rid2] = met
			return EmbeddingResponse(embeddings=embeddings, metrics=metrics_map)

		with ThreadPoolExecutor(max_workers=max_workers) as ex:
			futs = [ex.submit(self._embed_one, rid, text) for rid, text in pairs]
			for fut in as_completed(futs):
				rid2, emb, met = fut.result()
				embeddings[rid2] = emb
				metrics_map[rid2] = met

		return EmbeddingResponse(embeddings=embeddings, metrics=metrics_map)

	async def async_embed_texts(self, *args: Any, **kwargs: Any) -> Union[EmbeddingResponse, BatchJobResponse]:
		return await asyncio.to_thread(self.embed_texts, *args, **kwargs)

	# ------------------------------------------------------------------
	# Batch embeddings helpers
	# ------------------------------------------------------------------

	def submit_embedding_batch_job(
			self,
			pairs: Union[List[Tuple[str, str]], RecordInput],
			*,
			s3_bucket: Optional[str],
			s3_prefix: str,
			role_arn: Optional[str],
	) -> BatchJobResponse:
		if s3_bucket is None or role_arn is None:
			raise ValueError('s3_bucket and role_arn must be provided for batch embedding jobs')

		if not isinstance(pairs, list) or (pairs and not isinstance(pairs[0], tuple)):
			pairs = normalize_records(pairs)  # type: ignore[assignment]

		job_uuid = uuid.uuid4().hex
		job_name = f'bedrock-embedding-job-{job_uuid}'

		tmp_path: Optional[str] = None
		try:
			with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
				for record_id, text in pairs:
					entry = {'recordId': str(record_id), 'modelInput': {'inputText': text}}
					tmp.write(json.dumps(entry) + '\n')
				tmp_path = tmp.name

			input_key = f'{s3_prefix}/inputs/{job_uuid}.jsonl'
			self.s3.upload_file(tmp_path, s3_bucket, input_key)
			input_s3_uri = f's3://{s3_bucket}/{input_key}'

			output_prefix = f'{s3_prefix}/outputs/{job_uuid}/'
			output_s3_uri = f's3://{s3_bucket}/{output_prefix}'

			resp = self.bedrock.create_model_invocation_job(
				jobName=job_name,
				modelId=self.embedding_model_id,
				roleArn=role_arn,
				inputDataConfig={'s3InputDataConfig': {'s3Uri': input_s3_uri}},
				outputDataConfig={'s3OutputDataConfig': {'s3Uri': output_s3_uri}},
			)

			job_id = (
					resp.get("jobArn")
					or resp.get("jobId")
					or resp.get("jobIdentifier")
					or resp.get("id")
			)
			if not job_id:
				job_id = job_name

			return BatchJobResponse(
				job_id=str(job_id),
				job_name=job_name,
				model_id=self.embedding_model_id,
				input_s3_uri=input_s3_uri,
				output_s3_uri=output_s3_uri,
				response=resp,
			)
		finally:
			if tmp_path:
				try:
					os.remove(tmp_path)
				except OSError:
					pass

	def get_batch_job(self, job_id: str) -> dict:
		return self.bedrock.get_model_invocation_job(jobIdentifier=job_id)

	def wait_for_batch_job(
			self,
			job_id: str,
			*,
			poll_seconds: float = 10.0,
			timeout_seconds: float = 3600.0,
	) -> dict:

		deadline = time.time() + timeout_seconds
		while True:
			info = self.get_batch_job(job_id)
			status = info.get('status')
			if status in ('Completed', 'Failed', 'Stopped', 'Expired'):
				return info
			if time.time() > deadline:
				raise TimeoutError(f'Batch job did not complete within {timeout_seconds}s (status={status})')
			time.sleep(poll_seconds)

	def download_batch_results_jsonl(self, *, output_s3_uri: str) -> Iterator[dict]:
		if not output_s3_uri.startswith('s3://'):
			raise ValueError('output_s3_uri must start with s3://')

		rest = output_s3_uri[len('s3://'):]
		bucket, _, prefix = rest.partition('/')
		if prefix and not prefix.endswith('/'):
			prefix += '/'

		paginator = self.s3.get_paginator('list_objects_v2')
		for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
			for obj in page.get('Contents', []) or []:
				key = obj.get('Key')
				if not key or not key.endswith('.jsonl'):
					continue
				body = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
				for line in body.splitlines():
					line = line.strip()
					if not line:
						continue
					yield json.loads(line.decode('utf-8'))

	def parse_batch_embeddings(self, *, output_s3_uri: str) -> Dict[str, List[float]]:
		out: Dict[str, List[float]] = {}
		for obj in self.download_batch_results_jsonl(output_s3_uri=output_s3_uri):
			rid = obj.get('recordId') or obj.get('record_id')
			if rid is None:
				continue
			model_out = obj.get('modelOutput') or obj.get('model_output') or obj.get('output') or {}
			if isinstance(model_out, dict):
				emb = model_out.get('embedding')
				if isinstance(emb, list):
					out[str(rid)] = emb
		return out
