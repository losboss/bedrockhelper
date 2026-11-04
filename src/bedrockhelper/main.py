"""
A helper module for interacting with Amazon Bedrock for retrieval-augmented
generation (RAG) and for producing vector embeddings.

- RAG Q&A goes through the Bedrock Runtime Converse / ConverseStream APIs,
  which work across model providers. prefer_converse=False sends an Anthropic
  Messages body through invoke_model instead, for Claude models only.
- Embeddings via Titan embed text v2:
  - small batches: synchronous invoke_model (optionally concurrent)
  - large batches: S3 JSONL + create_model_invocation_job (batch inference)
- Credentials come from boto3's default resolution, optionally exchanged for an
  assumed role (see bedrockhelper.session).
- Sync methods, with async wrappers that run them in a worker thread.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

from botocore.config import Config

from bedrockhelper.models import BatchJobResponse, EmbeddingResponse, InvocationMetrics, RAGResponse, RAGStream
from bedrockhelper.session import BotoSessionManager, is_expired_token
from bedrockhelper.types import RecordInput
from bedrockhelper.utils import (
	UNSET,
	TemperatureArg,
	Unset,
	build_converse_request,
	extract_converse_text,
	format_context_passages,
	is_sampling_param_error,
	iter_stream_text,
	normalize_headers,
	normalize_records,
	normalize_s3_bucket_name,
	resolve_temperature,
	strip_sampling_params,
	truncate_by_chars,
)

_T = TypeVar('_T')

log = logging.getLogger(__name__)


def _to_int(value: Any, default: int | None = None) -> int | None:
	"""
	A metric as an int, or `default` when it is missing or not a number.

	Metrics are informational: one Bedrock reports in an unexpected form is
	left unset rather than failing a request that otherwise succeeded.
	"""
	if value is None:
		return default
	try:
		return int(value)
	except (TypeError, ValueError):
		return default


def _settle(future: asyncio.Future[_T], *, result: _T | None = None, error: BaseException | None = None) -> None:
	"""
	Complete `future` unless the caller already cancelled it.

	Scheduled with call_soon_threadsafe, so the done() check runs on the event
	loop: checking from the worker thread could race with a cancellation.
	"""
	if future.done():
		return
	if error is not None:
		future.set_exception(error)
	else:
		future.set_result(result)  # type: ignore[arg-type]


def _duration_from_environment() -> int:
	raw = os.getenv('AWS_ASSUME_ROLE_DURATION')
	if raw is None:
		return 3600
	try:
		return int(raw)
	except ValueError:
		raise ValueError(f'AWS_ASSUME_ROLE_DURATION must be a whole number of seconds, not {raw!r}') from None


class BedrockHelper:
	"""
	Utility class for interacting with Amazon Bedrock.

	Synchronous core methods; async wrappers provided.

	Important split:
	- generate_with_rag(...) -> non-streaming only (returns full text + metrics)
	- stream_with_rag(...)   -> streaming only (returns RAGStream which yields deltas + metrics)

	Settings left out are read from the environment when the helper is
	constructed: AWS_REGION, AWS_ROLE_ARN, AWS_ROLE_SESSION_NAME,
	AWS_EXTERNAL_ID, AWS_ASSUME_ROLE_DURATION, AWS_MODEL_ID and
	AWS_EMBEDDING_MODEL_ID. Passing role_arn=None (or external_id=None)
	explicitly means none, even when the variable is set.
	"""

	def __init__(
		self,
		region_name: str | None = None,
		rag_model_id: str | None = None,
		embedding_model_id: str | None = None,
		*,
		botocore_config: Config | None = None,
		role_arn: str | None | Unset = UNSET,
		role_session_name: str | None = None,
		external_id: str | None | Unset = UNSET,
		assume_role_duration_seconds: int | None = None,
		# injected clients (tests)
		s3_client: Any | None = None,
		bedrock_runtime_client: Any | None = None,
		bedrock_client: Any | None = None,
		# streaming
		max_concurrent_streams: int = 8,
	) -> None:
		self.rag_model_id = rag_model_id or os.environ.get('AWS_MODEL_ID') or 'global.anthropic.claude-sonnet-5'
		self.embedding_model_id = (
			embedding_model_id or os.environ.get('AWS_EMBEDDING_MODEL_ID') or 'amazon.titan-embed-text-v2:0'
		)
		if max_concurrent_streams < 1:
			raise ValueError('max_concurrent_streams must be >= 1')

		self._max_concurrent_streams = max_concurrent_streams
		self._stream_sema = threading.BoundedSemaphore(max_concurrent_streams)

		if botocore_config is None:
			retries: Any = {'max_attempts': 10, 'mode': 'adaptive'}
			botocore_config = Config(
				retries=retries,
				connect_timeout=5,
				read_timeout=120,
			)

		self._session_mgr = BotoSessionManager(
			region_name=region_name or os.getenv('AWS_REGION') or 'ca-central-1',
			botocore_config=botocore_config,
			role_arn=os.getenv('AWS_ROLE_ARN') if isinstance(role_arn, Unset) else role_arn,
			role_session_name=role_session_name or os.getenv('AWS_ROLE_SESSION_NAME') or 'bedrockhelper',
			external_id=os.getenv('AWS_EXTERNAL_ID') if isinstance(external_id, Unset) else external_id,
			assume_role_duration_seconds=(
				assume_role_duration_seconds
				if assume_role_duration_seconds is not None
				else _duration_from_environment()
			),
		)

		# Track which clients were injected; we won't overwrite them on refresh.
		self._injected = {
			'bedrock_runtime': bedrock_runtime_client is not None,
			'bedrock': bedrock_client is not None,
			's3': s3_client is not None,
		}

		self.bedrock_runtime = bedrock_runtime_client or self._session_mgr.client('bedrock-runtime')
		self.bedrock = bedrock_client or self._session_mgr.client('bedrock')
		self.s3 = s3_client or self._session_mgr.client('s3')

	def _refresh_clients_if_needed(self) -> None:
		# Rebind only non-injected clients.
		if not self._injected['bedrock_runtime']:
			self.bedrock_runtime = self._session_mgr.client('bedrock-runtime')
		if not self._injected['bedrock']:
			self.bedrock = self._session_mgr.client('bedrock')
		if not self._injected['s3']:
			self.s3 = self._session_mgr.client('s3')

	def _call_with_refresh(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
		try:
			return fn(*args, **kwargs)
		except Exception as e:
			if not is_expired_token(e):
				raise
			# Invalidate session+clients and retry once.
			self._session_mgr.invalidate()
			self._refresh_clients_if_needed()
			return fn(*args, **kwargs)

	def _call_converse_with_sampling_fallback(self, fn: Callable[..., _T], request: dict[str, Any]) -> _T:
		"""
		Invoke converse()/converse_stream(), retrying once without sampling
		parameters if the model rejects them.

		Covers models released after this version, and models that accept
		temperature alone but reject it alongside topP.
		"""
		try:
			return self._call_with_refresh(fn, **request)
		except Exception as e:
			if not is_sampling_param_error(e):
				raise
			retry_request, removed = strip_sampling_params(request)
			if not removed:
				raise
			log.warning(
				'Model %s rejected sampling parameters; retrying without them: %s',
				request.get('modelId'),
				e,
			)
			return self._call_with_refresh(fn, **retry_request)

	def _call_invoke_with_sampling_fallback(
		self,
		fn: Callable[..., _T],
		*,
		model_id: str,
		body: dict[str, Any],
		**kwargs: Any,
	) -> _T:
		"""
		Invoke invoke_model()/invoke_model_with_response_stream(), retrying once
		without sampling parameters if the model rejects them.
		"""
		try:
			return self._call_with_refresh(fn, modelId=model_id, body=json.dumps(body), **kwargs)
		except Exception as e:
			if not is_sampling_param_error(e):
				raise
			retry_body, removed = strip_sampling_params(body)
			if not removed:
				raise
			log.warning(
				'Model %s rejected sampling parameters; retrying without them: %s',
				model_id,
				e,
			)
			return self._call_with_refresh(fn, modelId=model_id, body=json.dumps(retry_body), **kwargs)

	# ------------------------------------------------------------------
	# Metrics helpers
	# ------------------------------------------------------------------

	@classmethod
	def _extract_metrics_from_response(
		cls,
		response: dict[str, Any],
		response_body: dict[str, Any] | None = None,
	) -> InvocationMetrics:
		metrics = InvocationMetrics()

		# ---- Body metrics (model-specific)
		if response_body and isinstance(response_body, dict):
			usage = response_body.get('usage')
			if isinstance(usage, dict):
				metrics.usage = usage

				def _pick_int(*keys: str) -> int | None:
					return next((n for n in (_to_int(usage.get(k)) for k in keys) if n is not None), None)

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

			# Converse responses, and the metadata event that ends a ConverseStream.
			converse_metrics = response_body.get('metrics')
			if isinstance(converse_metrics, dict):
				metrics.invocation_latency_ms = _to_int(converse_metrics.get('latencyMs'))

			# Titan embed frequently uses inputTextTokenCount rather than a usage dict
			if metrics.input_tokens is None:
				metrics.input_tokens = _to_int(response_body.get('inputTextTokenCount'))

			# Bedrock's own counts (invoke_model streams) take precedence where present.
			inv = response_body.get('amazon-bedrock-invocationMetrics')
			if isinstance(inv, dict):
				metrics.input_tokens = _to_int(inv.get('inputTokenCount'), metrics.input_tokens)
				metrics.output_tokens = _to_int(inv.get('outputTokenCount'), metrics.output_tokens)
				metrics.invocation_latency_ms = _to_int(inv.get('invocationLatency'), metrics.invocation_latency_ms)
				metrics.first_byte_latency_ms = _to_int(inv.get('firstByteLatency'), metrics.first_byte_latency_ms)

		# ---- Header metrics
		http_headers = response.get('ResponseMetadata', {}).get('HTTPHeaders', {}) or {}
		if http_headers:
			norm = normalize_headers(http_headers)
			metrics.header_metrics = dict(norm)

			if metrics.input_tokens is None:
				metrics.input_tokens = _to_int(norm.get('x-amzn-bedrock-input-token-count'))
			if metrics.output_tokens is None:
				metrics.output_tokens = _to_int(norm.get('x-amzn-bedrock-output-token-count'))
			if metrics.invocation_latency_ms is None:
				metrics.invocation_latency_ms = _to_int(norm.get('x-amzn-bedrock-invocation-latency'))
			if metrics.first_byte_latency_ms is None:
				metrics.first_byte_latency_ms = _to_int(norm.get('x-amzn-bedrock-first-byte-latency'))

		if metrics.total_tokens is None and metrics.input_tokens is not None and metrics.output_tokens is not None:
			metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

		return metrics

	# ------------------------------------------------------------------
	# RAG
	# ------------------------------------------------------------------

	@staticmethod
	def _prepare_context(
		context: str | Sequence[str],
		*,
		include_headers: bool,
		max_chars: int | None,
	) -> str:
		if isinstance(context, str):
			ctx = context
		else:
			ctx = format_context_passages(list(context), include_headers=include_headers)
		return truncate_by_chars(ctx, max_chars)

	@staticmethod
	def _claude_messages_body(
		*,
		model_id: str,
		system_prompt: str,
		context: str,
		question: str,
		temperature: TemperatureArg,
		max_tokens: int,
		rag_instructions: str,
		extra_params: dict[str, Any],
	) -> dict[str, Any]:
		"""Anthropic Messages request body for invoke_model(); Claude models only."""
		user_text = f'{rag_instructions}\nContext:\n{context}\n'
		if question:
			user_text += f'\nQuestion:\n{question}\n'

		body: dict[str, Any] = {
			'anthropic_version': 'bedrock-2023-05-31',
			'system': system_prompt,
			'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': user_text}]}],
			'max_tokens': max_tokens,
		}
		resolved_temperature = resolve_temperature(model_id, temperature)
		if resolved_temperature is not None:
			body['temperature'] = resolved_temperature
		body.update(extra_params)
		return body

	def generate_with_rag(
		self,
		*,
		system_prompt: str,
		context: str | Sequence[str],
		include_headers_in_context: bool = False,
		question: str | None = '',
		model_id: str | None = None,
		temperature: TemperatureArg = UNSET,
		max_tokens: int = 8192,
		max_context_chars: int | None = 150_000,
		prefer_converse: bool = True,
		rag_instructions: str = '',
		**extra_params: Any,
	) -> RAGResponse:
		"""
		Answer `question` from `context` and return the full response.

		Uses the Converse API. With prefer_converse=False it sends an Anthropic
		Messages body through invoke_model instead, which only Claude models
		accept. Errors from Bedrock are raised as they are; the only retry is a
		single one without sampling parameters when a model rejects them.
		"""
		selected_model = model_id or self.rag_model_id
		ctx = self._prepare_context(context, include_headers=include_headers_in_context, max_chars=max_context_chars)

		if prefer_converse:
			req = build_converse_request(
				model_id=selected_model,
				system_prompt=system_prompt,
				context=ctx,
				question=str(question or ''),
				max_tokens=max_tokens,
				temperature=temperature,
				rag_instructions=rag_instructions,
				**extra_params,
			)
			resp = self._call_converse_with_sampling_fallback(self.bedrock_runtime.converse, req)
			text = extract_converse_text(resp)
			metrics = self._extract_metrics_from_response(resp, resp)
			return RAGResponse(text=text, stream=False, metrics=metrics, raw_response=resp)

		body = self._claude_messages_body(
			model_id=selected_model,
			system_prompt=system_prompt,
			context=ctx,
			question=str(question or ''),
			temperature=temperature,
			max_tokens=max_tokens,
			rag_instructions=rag_instructions,
			extra_params=extra_params,
		)
		resp = self._call_invoke_with_sampling_fallback(
			self.bedrock_runtime.invoke_model,
			model_id=selected_model,
			body=body,
		)
		body_json = json.loads(resp['body'].read())
		text = self._extract_text_from_claude(body_json)
		metrics = self._extract_metrics_from_response(resp, body_json if isinstance(body_json, dict) else None)
		return RAGResponse(text=text, stream=False, metrics=metrics, raw_response=body_json)

	@staticmethod
	def _extract_text_from_claude(body: Any) -> str:
		if isinstance(body, dict):
			content = body.get('content')
			if isinstance(content, list):
				parts: list[str] = []
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

	def stream_with_rag(
		self,
		*,
		system_prompt: str,
		context: str | Sequence[str],
		include_headers_in_context: bool = False,
		question: str | None = '',
		model_id: str | None = None,
		temperature: TemperatureArg = UNSET,
		max_tokens: int = 8192,
		max_context_chars: int | None = 150_000,
		prefer_converse: bool = True,
		rag_instructions: str = '',
		**extra_params: Any,
	) -> RAGStream:
		"""
		Streaming counterpart of generate_with_rag(), with the same arguments.

		Iterate the returned RAGStream for text deltas; afterwards `.result`
		holds the full text and the token counts Bedrock reported at the end.
		"""
		selected_model = model_id or self.rag_model_id
		ctx = self._prepare_context(context, include_headers=include_headers_in_context, max_chars=max_context_chars)

		if prefer_converse:
			req = build_converse_request(
				model_id=selected_model,
				system_prompt=system_prompt,
				context=ctx,
				question=str(question or ''),
				max_tokens=max_tokens,
				temperature=temperature,
				rag_instructions=rag_instructions,
				**extra_params,
			)
			resp = self._call_converse_with_sampling_fallback(self.bedrock_runtime.converse_stream, req)
			return self._make_stream(resp, resp['stream'])

		body = self._claude_messages_body(
			model_id=selected_model,
			system_prompt=system_prompt,
			context=ctx,
			question=str(question or ''),
			temperature=temperature,
			max_tokens=max_tokens,
			rag_instructions=rag_instructions,
			extra_params=extra_params,
		)
		resp = self._call_invoke_with_sampling_fallback(
			self.bedrock_runtime.invoke_model_with_response_stream,
			model_id=selected_model,
			body=body,
		)
		return self._make_stream(resp, resp['body'])

	def _make_stream(self, resp: dict[str, Any], event_stream: Iterable[dict[str, Any]]) -> RAGStream:
		"""Wrap a Bedrock event stream; the end-of-stream metrics land in `.result.metrics`."""
		tail: dict[str, Any] = {}

		def iterator_factory() -> Iterator[str]:
			try:
				yield from iter_stream_text(event_stream, on_metrics=tail.update)
			finally:
				close = getattr(event_stream, 'close', None)
				if callable(close):
					close()

		def build_final(full_text: str) -> RAGResponse:
			return RAGResponse(
				text=full_text,
				stream=True,
				metrics=self._extract_metrics_from_response(resp, tail),
				raw_response=tail or None,
			)

		return RAGStream(
			iterator_factory=iterator_factory,
			header_metrics=self._extract_metrics_from_response(resp, None),
			build_final=build_final,
		)

	# ------------------------------------------------------------------
	# Async wrappers for RAG
	# ------------------------------------------------------------------

	async def async_generate_with_rag(self, **kwargs: Any) -> RAGResponse:
		return await asyncio.to_thread(self.generate_with_rag, **kwargs)

	async def async_stream_with_rag(
		self,
		**kwargs: Any,
	) -> tuple[AsyncIterator[str], InvocationMetrics, asyncio.Future[RAGResponse]]:
		"""
		Returns:
		  (async_iterator_of_text, header_metrics, result_future)

		- The iterator yields deltas for live streaming, and raises if the
		  stream fails part-way.
		- header_metrics are available as soon as Bedrock has accepted the request.
		- result_future resolves to the final RAGResponse (full text and the
		  end-of-stream token counts) once the iterator is exhausted.

		An error before the stream starts (bad request, access denied) is raised
		from this call itself. At most max_concurrent_streams run at once.
		"""
		loop = asyncio.get_running_loop()
		# Items: a text delta, an exception that ended the stream, or None at the end.
		q: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
		header_future: asyncio.Future[InvocationMetrics] = loop.create_future()
		result_future: asyncio.Future[RAGResponse] = loop.create_future()

		def worker() -> None:
			with self._stream_sema:
				try:
					stream = self.stream_with_rag(**kwargs)
				# BaseException: whatever escapes here would leave the caller
				# awaiting header_future forever.
				except BaseException as e:
					loop.call_soon_threadsafe(functools.partial(_settle, header_future, error=e))
					return
				loop.call_soon_threadsafe(functools.partial(_settle, header_future, result=stream.header_metrics))
				try:
					for t in stream:
						loop.call_soon_threadsafe(q.put_nowait, t)
				# BaseException, as above: it is handed to the caller's iterator and
				# result_future, and an escape would leave the iterator waiting forever.
				except BaseException as e:
					loop.call_soon_threadsafe(functools.partial(_settle, result_future, error=e))
					loop.call_soon_threadsafe(q.put_nowait, e)
					return
				loop.call_soon_threadsafe(functools.partial(_settle, result_future, result=stream.result))
				loop.call_soon_threadsafe(q.put_nowait, None)

		threading.Thread(target=worker, daemon=True).start()
		header_metrics = await header_future

		async def chunks() -> AsyncIterator[str]:
			while True:
				item = await q.get()
				if item is None:
					return
				if isinstance(item, BaseException):
					raise item
				yield item

		return chunks(), header_metrics, result_future

	# ------------------------------------------------------------------
	# Embeddings
	# ------------------------------------------------------------------

	def _embed_one(self, record_id: str, text: str) -> tuple[str, list[float], InvocationMetrics]:
		req = {'inputText': text}
		resp = self._call_with_refresh(
			self.bedrock_runtime.invoke_model,
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
		s3_bucket: str | None = None,
		s3_prefix: str | None = 'bedrock_batch',
		role_arn: str | None = None,
		max_workers: int | None = None,
		s3_bucket_owner: str | None = None,
	) -> EmbeddingResponse | BatchJobResponse:
		pairs = normalize_records(records)
		if not pairs:
			return EmbeddingResponse(embeddings={}, metrics={})

		if max_workers is None or max_workers <= 1 or len(pairs) == 1:
			max_workers = 1

		if len(pairs) <= batch_threshold:
			return self._embed_texts_sync_concurrent(pairs, max_workers=max_workers)

		return self.submit_embedding_batch_job(
			pairs,
			s3_bucket=s3_bucket,
			s3_prefix=s3_prefix or 'bedrock_batch',
			role_arn=role_arn,
			s3_bucket_owner=s3_bucket_owner,
		)

	def _embed_texts_sync_concurrent(
		self,
		pairs: list[tuple[str, str]],
		*,
		max_workers: int,
	) -> EmbeddingResponse:
		embeddings: dict[str, list[float]] = {}
		metrics_map: dict[str, InvocationMetrics] = {}

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

	async def async_embed_texts(self, *args: Any, **kwargs: Any) -> EmbeddingResponse | BatchJobResponse:
		return await asyncio.to_thread(self.embed_texts, *args, **kwargs)

	# ------------------------------------------------------------------
	# Batch embeddings helpers
	# ------------------------------------------------------------------

	def submit_embedding_batch_job(
		self,
		pairs: list[tuple[str, str]] | RecordInput,
		*,
		s3_bucket: str | None,
		s3_prefix: str,
		role_arn: str | None,
		s3_bucket_owner: str | None = None,
	) -> BatchJobResponse:
		if s3_bucket is None:
			raise ValueError('s3_bucket must be provided for batch embedding jobs')
		if role_arn is None:
			raise ValueError('role_arn must be provided for batch embedding jobs')

		s3_bucket = normalize_s3_bucket_name(s3_bucket)

		if not isinstance(pairs, list) or (pairs and not isinstance(pairs[0], tuple)):
			pairs = normalize_records(pairs)

		job_uuid = uuid.uuid4().hex
		job_name = f'bedrock-embedding-job-{job_uuid}'

		tmp_path: str | None = None
		try:
			with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
				for record_id, text in pairs:
					entry = {'recordId': str(record_id), 'modelInput': {'inputText': text}}
					tmp.write(json.dumps(entry) + '\n')
				tmp_path = tmp.name

			input_key = f'{s3_prefix}/inputs/{job_uuid}.jsonl'
			log.info(f'Uploading batch embedding input to s3://{s3_bucket}/{input_key}')
			self._call_with_refresh(self.s3.upload_file, tmp_path, s3_bucket, input_key)
			input_s3_uri = f's3://{s3_bucket}/{input_key}'

			output_prefix = f'{s3_prefix}/outputs/{job_uuid}/'
			output_s3_uri = f's3://{s3_bucket}/{output_prefix}'

			input_config: dict[str, Any] = {'s3InputDataConfig': {'s3Uri': input_s3_uri}}
			output_config: dict[str, Any] = {'s3OutputDataConfig': {'s3Uri': output_s3_uri}}

			if s3_bucket_owner is not None:
				input_config['s3InputDataConfig']['s3BucketOwner'] = str(s3_bucket_owner)
				output_config['s3OutputDataConfig']['s3BucketOwner'] = str(s3_bucket_owner)

			log.info(f'Creating batch embedding job {job_name}')
			resp = self._call_with_refresh(
				self.bedrock.create_model_invocation_job,
				jobName=job_name,
				modelId=self.embedding_model_id,
				roleArn=role_arn,
				inputDataConfig=input_config,
				outputDataConfig=output_config,
			)

			job_id = resp.get('jobArn') or resp.get('jobId') or resp.get('jobIdentifier') or resp.get('id')
			if not job_id:
				job_id = job_name

			log.info(f'Created batch embedding job {job_name} with ID {job_id}')
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
				except OSError as e:
					# Raising from this finally block would replace the upload or job
					# error the caller needs; a leftover temp file is only clutter.
					log.warning('Could not delete batch input file %s: %s', tmp_path, e)

	def get_batch_job(self, job_id: str) -> dict[str, Any]:
		return self._call_with_refresh(self.bedrock.get_model_invocation_job, jobIdentifier=job_id)

	def wait_for_batch_job(
		self,
		job_id: str,
		*,
		poll_seconds: float = 10.0,
		timeout_seconds: float = 3600.0,
	) -> dict[str, Any]:
		deadline = time.time() + timeout_seconds
		while True:
			info = self.get_batch_job(job_id)
			status = info.get('status')
			if status in ('Completed', 'Failed', 'Stopped', 'Expired'):
				return info
			if time.time() > deadline:
				raise TimeoutError(f'Batch job did not complete within {timeout_seconds}s (status={status})')
			time.sleep(poll_seconds)

	def download_batch_results_jsonl(self, *, output_s3_uri: str) -> Iterator[dict[str, Any]]:
		if not output_s3_uri.startswith('s3://'):
			raise ValueError('output_s3_uri must start with s3://')

		rest = output_s3_uri[len('s3://') :]
		bucket, _, prefix = rest.partition('/')
		if prefix and not prefix.endswith('/'):
			prefix += '/'

		paginator = self._call_with_refresh(self.s3.get_paginator, 'list_objects_v2')
		for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
			for obj in page.get('Contents', []) or []:
				key = obj.get('Key')
				if not key or not key.endswith(('.jsonl', '.jsonl.out')):
					continue
				resp = self._call_with_refresh(self.s3.get_object, Bucket=bucket, Key=key)
				body = resp['Body'].read()
				for line in body.splitlines():
					line = line.strip()
					if not line:
						continue
					yield json.loads(line.decode('utf-8'))

	def parse_batch_embeddings(self, *, output_s3_uri: str) -> dict[str, list[float]]:
		out: dict[str, list[float]] = {}
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
