import ast
import json
import threading
from queue import Queue
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from bedrockhelper.types import RecordInput

DEFAULT_STOP_TYPES_INVOKE: Tuple[str, ...] = ('message_stop', 'completion_stop', 'end')
DEFAULT_STOP_KEYS_CONVERSE: Tuple[str, ...] = ('messageStop', 'message_stop', 'stop', 'end')

_SENTINEL = object()


def iter_bedrock_stream_text(
	event_stream: Iterable[dict],
	*,
	on_text: Callable[[str], None],
	stream_kind: str = 'auto',
	invoke_stop_types: Tuple[str, ...] = DEFAULT_STOP_TYPES_INVOKE,
	converse_stop_keys: Tuple[str, ...] = DEFAULT_STOP_KEYS_CONVERSE,
	decode: str = 'utf-8',
) -> None:
	"""
	Iterate a Bedrock streaming event stream and call `on_text(...)` for each text delta.

	(docstring trimmed for brevity)
	"""

	def _handle_invoke_event(event: dict) -> bool:
		# Returns True if should stop
		chunk = event.get('chunk')
		if not chunk:
			return False

		raw = chunk.get('bytes')
		if not raw:
			return False

		try:
			msg = json.loads(raw.decode(decode))
		except Exception:
			return False
		if not isinstance(msg, dict):
			return False
		delta = msg.get('delta')
		if isinstance(delta, dict):
			t = delta.get('text')
			if isinstance(t, str) and t:
				on_text(t)

		if msg.get('type') in invoke_stop_types:
			return True
		return False

	def _handle_converse_event(event: dict) -> bool:
		# Returns True if should stop
		cbd = event.get('contentBlockDelta') or event.get('content_block_delta')
		if isinstance(cbd, dict):
			delta = cbd.get('delta')
			if isinstance(delta, dict):
				t = delta.get('text')
				if isinstance(t, str) and t:
					on_text(t)

		for k in converse_stop_keys:
			if k in event and event.get(k):
				return True

		ms = event.get('messageStop') or event.get('message_stop')
		if isinstance(ms, dict) and ms:
			return True

		return False

	for event_item in event_stream:
		if not isinstance(event_item, dict):
			continue

		kind = stream_kind

		if kind == 'auto':
			# Prefer converse when a converse-like shape or stop key is present.
			has_converse = False
			if (
				'contentBlockDelta' in event_item
				or 'content_block_delta' in event_item
				or 'messageStop' in event_item
				or 'message_stop' in event_item
			):
				has_converse = True
			else:
				for k in converse_stop_keys:
					if k in event_item and event_item.get(k):
						has_converse = True
						break

			if has_converse:
				should_stop = _handle_converse_event(event_item)
				if should_stop:
					break
				# Do not also treat the same event as an invoke event.
				continue

			# If no converse signal, but a chunk is present, treat as invoke.
			if 'chunk' in event_item:
				kind = 'invoke'
			else:
				# nothing recognizable in auto mode; skip
				continue

		if kind == 'invoke':
			should_stop = _handle_invoke_event(event_item)
			if should_stop:
				break
		else:  # explicit "converse"
			should_stop = _handle_converse_event(event_item)
			if should_stop:
				break


def normalize_headers(headers: Mapping[str | int, Any]) -> Dict[str, Any]:
	return {str(k).lower(): v for k, v in headers.items()}


def format_context_passages(
	passages: Sequence[str],
	*,
	include_headers: bool = False,
	header_template: str = '--- passage {i} ---',
	separator: str = '\n\n',
) -> str:
	blocks: List[str] = []
	for i, p in enumerate(passages, start=1):
		text = str(p)
		if include_headers:
			blocks.append(f'{header_template.format(i=i)}\n{text}')
		else:
			blocks.append(text)
	return separator.join(blocks)


def truncate_by_chars(text: str, max_chars: Optional[int]) -> str:
	if max_chars is None:
		return text
	if max_chars <= 0:
		return ''
	return text[:max_chars]


def normalize_records(records: RecordInput) -> List[Tuple[str, str]]:
	if isinstance(records, dict):
		return [(str(k), v) for k, v in records.items()]

	out: List[Tuple[str, str]] = []
	for item in records:
		if isinstance(item, tuple) and len(item) == 2:
			out.append((str(item[0]), item[1]))
			continue
		if isinstance(item, Mapping):
			if len(item) != 1:
				raise ValueError(f'Each mapping record must have exactly 1 entry; got {len(item)}')
			k, v = next(iter(item.items()))
			out.append((str(k), v))
			continue
		raise ValueError(f'Unsupported record item type: {type(item)}')
	return out


def extract_converse_text(resp: dict) -> str:
	"""
	Extract final assistant text from a Bedrock converse() response.
	"""
	try:
		content = resp.get('output', {}).get('message', {}).get('content', [])
		parts: List[str] = []
		for item in content:
			if isinstance(item, dict):
				t = item.get('text')
				if isinstance(t, str):
					parts.append(t)
		return ''.join(parts)
	except Exception:
		return ''


def build_converse_request(
	model_id: str,
	system_prompt: str,
	context: str,
	question: str,
	max_tokens: int,
	temperature: float,
	rag_instructions: str = '',
	**extra_params: Any,
) -> Dict[str, Any]:
	"""
	Build the shared request payload for converse() and converse_stream().
	"""

	user_text = f'Context:\n{context}\n'
	if question:
		user_text += f'\nQuestion:\n{question}\n'
	if rag_instructions:
		user_text = f'{rag_instructions}\n{user_text}'

	req: Dict[str, Any] = {
		'modelId': model_id,
		'system': [{'text': system_prompt}],
		'messages': [
			{
				'role': 'user',
				'content': [{'text': user_text}],
			}
		],
		'inferenceConfig': {
			'maxTokens': max_tokens,
			'temperature': temperature,
		},
		# Enable usage metrics in streaming responses
		'performanceConfig': {
			'latency': 'standard'  # or 'optimized'
		},
		# Request additional metadata
		'additionalModelRequestFields': {},
		'additionalModelResponseFieldPaths': ['/usage'],
	}

	# Allow caller to pass model-specific / top-level extra fields.
	# (If a key collides, extra_params wins.)
	if extra_params:
		req.update(extra_params)

	return req


def normalize_s3_bucket_name(bucket: str) -> str:
	"""
	Strips "s3://" prefix from bucket name if present.
	"""
	bucket = bucket.strip()
	if bucket.startswith('s3://'):
		return bucket[len('s3://') :]
	return bucket


def _decode_bedrock_chunk_bytes(raw: Any, *, decode: str = 'utf-8') -> Optional[str]:
	"""
	Bedrock streaming 'chunk.bytes' is usually bytes, but can sometimes appear as:
	- a str that is already JSON, or
	- a str that looks like a Python bytes literal: "b'...'"
	Return decoded text or None if we can't decode.
	"""
	if raw is None:
		return None

	# Normal case: bytes
	if isinstance(raw, (bytes, bytearray)):
		return bytes(raw).decode(decode, errors='replace')

	# Sometimes: already-decoded JSON string
	if isinstance(raw, str):
		s = raw.strip()

		# If it looks like b'...'
		if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
			try:
				b = ast.literal_eval(s)  # -> bytes
				if isinstance(b, (bytes, bytearray)):
					return bytes(b).decode(decode, errors='replace')
			except Exception:
				return None

		# Otherwise assume it's JSON text already
		return s

	return None


def _iter_stream_with_callback(event_stream, *, stream_kind: str) -> Iterator[str]:
	"""
	Adapt iter_bedrock_stream_text(callback-based) into an Iterator[str].
	Ensures the underlying stream is closed when the iterator ends.
	"""
	q: 'Queue[object]' = Queue()
	err: Dict[str, BaseException] = {}

	def on_text(t: str) -> None:
		q.put(t)

	def worker() -> None:
		try:
			iter_bedrock_stream_text(event_stream, on_text=on_text, stream_kind=stream_kind)
		except BaseException as e:
			err['exc'] = e
		finally:
			try:
				if hasattr(event_stream, 'close'):
					event_stream.close()
			finally:
				q.put(_SENTINEL)

	threading.Thread(target=worker, daemon=True).start()

	while True:
		item = q.get()
		if item is _SENTINEL:
			break
		yield item  # str

	if 'exc' in err:
		raise err['exc']


def iter_bedrock_stream_text_gen_with_tail(
	event_stream: Iterable[dict],
	*,
	decode: str = 'utf-8',
	on_tail: Callable[[dict], None],
) -> Iterator[str]:
	"""
	Yield text deltas. Best-effort: whenever we parse a JSON message that contains
	usage/metrics-ish fields, we call on_tail(msg).
	"""

	def _maybe_capture_tail(msg: Any) -> None:
		if not isinstance(msg, dict):
			return
		# best-effort heuristics
		if (
			'usage' in msg
			or 'amazon-bedrock-invocationMetrics' in msg
			or 'invocationMetrics' in msg
			or 'messageStop' in msg
			or 'message_stop' in msg
		):
			on_tail(msg)

	for event_item in event_stream:
		if not isinstance(event_item, dict):
			continue

		if 'messageStop' in event_item or 'message_stop' in event_item:
			_maybe_capture_tail(event_item)
			break

		# INVOKE shape: {"chunk": {"bytes": b"...json..."}}
		if 'chunk' in event_item and isinstance(event_item.get('chunk'), dict):
			raw = event_item['chunk'].get('bytes')
			text = _decode_bedrock_chunk_bytes(raw, decode=decode)
			if not text:
				continue
			try:
				msg = json.loads(text)
			except Exception:
				continue

			_maybe_capture_tail(msg)

			if not isinstance(msg, dict):
				continue
			delta = msg.get('delta')
			if isinstance(delta, dict):
				t = delta.get('text')
				if isinstance(t, str) and t:
					yield t

			# stop condition
			if msg.get('type') in ('message_stop', 'messageStop', 'stop', 'end_turn', 'endTurn'):
				break

		# CONVERSE-like shape: {"contentBlockDelta": {"delta": {"text": "..."}}} etc.
		if 'contentBlockDelta' in event_item or 'content_block_delta' in event_item:
			_maybe_capture_tail(event_item)
			cbd = event_item.get('contentBlockDelta') or event_item.get('content_block_delta')
			if isinstance(cbd, dict):
				delta = cbd.get('delta')
				if isinstance(delta, dict):
					t = delta.get('text')
					if isinstance(t, str) and t:
						yield t
