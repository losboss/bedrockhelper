from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

_TERMINAL_STATUSES = {'Completed', 'Failed', 'Stopped', 'Expired'}
_SUCCESS_STATUSES = {'Completed'}
_FAILURE_STATUSES = {'Failed', 'Stopped', 'Expired'}


@dataclass
class InvocationMetrics:
	"""
	Encapsulates optional metrics returned by a Bedrock invocation.

	Metrics can appear:
	- in JSON body (model-specific)
	- in HTTP headers (x-amzn-bedrock-*)

	Missing fields remain None.
	"""

	input_tokens: Optional[int] = None
	output_tokens: Optional[int] = None
	total_tokens: Optional[int] = None
	invocation_latency_ms: Optional[int] = None
	first_byte_latency_ms: Optional[int] = None
	usage: Optional[Dict[str, Union[int, str]]] = None
	header_metrics: Optional[Dict[str, Union[str, int]]] = None


@dataclass
class RAGResponse:
	"""Return value for a RAG invocation."""

	text: str
	stream: bool
	metrics: InvocationMetrics
	raw_response: Optional[dict] = None


class RAGStream:
	"""
	Iterable stream of text deltas. After iteration completes, `.result`
	contains a RAGResponse with full text + best-effort end metrics.
	"""

	def __init__(
		self,
		*,
		iterator_factory: Callable[[], Iterator[str]],
		header_metrics: InvocationMetrics,
		build_final: Callable[[str, Optional[dict]], RAGResponse],
	) -> None:
		self._iterator_factory = iterator_factory
		self.header_metrics = header_metrics
		self._build_final = build_final

		self._chunks: List[str] = []
		self._tail_body: Optional[dict] = None
		self._result: Optional[RAGResponse] = None
		self._consumed = False

	@property
	def metrics(self) -> InvocationMetrics:
		"""
		Back-compat alias. This is the "early" metrics (mostly headers).
		Final/complete metrics are on `self.result.metrics`.
		"""
		return self.header_metrics

	def __iter__(self) -> Iterator[str]:
		if self._consumed:
			raise RuntimeError('RAGStream can only be iterated once')

		self._consumed = True

		for t in self._iterator_factory():
			self._chunks.append(t)
			yield t

		full_text = ''.join(self._chunks)
		self._result = self._build_final(full_text, self._tail_body)

	@property
	def result(self) -> RAGResponse:
		if self._result is None:
			raise RuntimeError('Stream not finished yet; iterate it to completion first')
		return self._result

	# internal hook
	def _set_tail_body(self, tail: dict) -> None:
		self._tail_body = tail


@dataclass
class EmbeddingResponse:
	"""Return value for synchronous embedding generation."""

	embeddings: Dict[str, List[float]]
	metrics: Dict[str, InvocationMetrics]


@dataclass(frozen=True, slots=True)
class BatchJobResponse:
	job_id: str  # IMPORTANT: should be the real Bedrock identifier
	job_name: str  # User-defined name
	model_id: str
	input_s3_uri: str
	output_s3_uri: str
	response: Dict[str, Any] = field(default_factory=dict)

	@staticmethod
	def status(info: Dict[str, Any]) -> Optional[str]:
		s = info.get('status')
		return s if isinstance(s, str) else None

	@staticmethod
	def is_done(info: Dict[str, Any]) -> bool:
		s = BatchJobResponse.status(info)
		return s in _TERMINAL_STATUSES

	@staticmethod
	def is_success(info: Dict[str, Any]) -> bool:
		s = BatchJobResponse.status(info)
		return s in _SUCCESS_STATUSES

	@staticmethod
	def is_failure(info: Dict[str, Any]) -> bool:
		s = BatchJobResponse.status(info)
		return s in _FAILURE_STATUSES

	@staticmethod
	def failure_reason(info: Dict[str, Any]) -> Optional[str]:
		"""
		Best-effort extraction of a human-readable failure reason/message.
		Bedrock payload shapes can vary, so we probe a few common keys.
		"""
		for key_path in (
			('failureMessage',),
			('message',),
			('errorMessage',),
			('error', 'message'),
			('failureDetails', 'message'),
			('failureReason',),
		):
			cur: Any = info
			ok = True
			for k in key_path:
				if not isinstance(cur, dict) or k not in cur:
					ok = False
					break
				cur = cur[k]
			if ok and isinstance(cur, str) and cur.strip():
				return cur.strip()
		return None

	@staticmethod
	def summary(info: Dict[str, Any]) -> str:
		s = BatchJobResponse.status(info) or 'Unknown'
		reason = BatchJobResponse.failure_reason(info)
		return f'{s}' + (f': {reason}' if reason else '')

	def to_ref(self) -> 'BatchJobRef':
		from bedrockhelper.batch.models import BatchJobRef

		return BatchJobRef(
			job_id=self.job_id,
			job_name=self.job_name,
			model_id=self.model_id,
			input_s3_uri=self.input_s3_uri,
			output_s3_uri=self.output_s3_uri,
		)
