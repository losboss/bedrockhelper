import json
from typing import Mapping, Dict, List, Sequence, Any, Callable, Iterable, Optional, Tuple

from bedrockhelper.types import RecordInput


DEFAULT_STOP_TYPES_INVOKE: Tuple[str, ...] = ("message_stop", "completion_stop", "end")
DEFAULT_STOP_KEYS_CONVERSE: Tuple[str, ...] = ("messageStop", "message_stop", "stop", "end")


def iter_bedrock_stream_text(
		event_stream: Iterable[dict],
		*,
		on_text: Callable[[str], None],
		stream_kind: str = "auto",
		invoke_stop_types: Tuple[str, ...] = DEFAULT_STOP_TYPES_INVOKE,
		converse_stop_keys: Tuple[str, ...] = DEFAULT_STOP_KEYS_CONVERSE,
		decode: str = "utf-8",
) -> None:
	"""
	Iterate a Bedrock streaming event stream and call `on_text(...)` for each text delta.

	Supports two Bedrock streaming shapes:
	  1) invoke_model_with_response_stream (Anthropic-style event JSON in chunk['bytes'])
		 - event: {"chunk": {"bytes": b"...json..."}}
		 - message delta: msg["delta"]["text"]
		 - stop: msg["type"] in invoke_stop_types

	  2) converse_stream (Bedrock Converse event dicts)
		 - event often contains: {"contentBlockDelta": {"delta": {"text": "..."}}}
		   (some SDKs may snake_case: "content_block_delta")
		 - stop events may include keys in converse_stop_keys set to truthy
		 - some events may include {"messageStop": {...}} (dict), or {"messageStop": True}

	Parameters
	----------
	event_stream:
		Iterable of events from the streaming call.
	on_text:
		Callback invoked for each text chunk/delta.
	stream_kind:
		"auto" (default), "invoke", or "converse".
		- auto: detect per-event; works even if a stream has occasional odd events.
	invoke_stop_types:
		Stop values for msg["type"] in invoke streams.
	converse_stop_keys:
		Keys that indicate stop in converse streams.
	decode:
		Byte decoding used for invoke streams.
	"""

	def _handle_invoke_event(event: dict) -> bool:
		# Returns True if should stop
		chunk = event.get("chunk")
		if not chunk:
			return False

		raw = chunk.get("bytes")
		if not raw:
			return False

		try:
			msg = json.loads(raw.decode(decode))
		except Exception:
			return False

		delta = msg.get("delta")
		if isinstance(delta, dict):
			t = delta.get("text")
			if isinstance(t, str) and t:
				on_text(t)

		if msg.get("type") in invoke_stop_types:
			return True
		return False

	def _handle_converse_event(event: dict) -> bool:
		# Returns True if should stop
		# text delta: event["contentBlockDelta"]["delta"]["text"] (or snake_case variant)
		cbd = event.get("contentBlockDelta") or event.get("content_block_delta")
		if isinstance(cbd, dict):
			delta = cbd.get("delta")
			if isinstance(delta, dict):
				t = delta.get("text")
				if isinstance(t, str) and t:
					on_text(t)

		# stop signals can appear as presence/truthiness of known keys
		for k in converse_stop_keys:
			if k in event and event.get(k):
				return True

		# Some shapes embed stop under a dict
		# e.g. {"messageStop": {"stopReason": "..."}}
		ms = event.get("messageStop") or event.get("message_stop")
		if isinstance(ms, dict) and ms:
			return True

		return False

	for event_item in event_stream:
		if not isinstance(event_item, dict):
			continue

		if stream_kind == "invoke":
			if _handle_invoke_event(event_item):
				break
			continue

		if stream_kind == "converse":
			if _handle_converse_event(event_item):
				break
			continue

		# auto-detect: prefer converse shape if it looks like converse
		if "contentBlockDelta" in event_item or "content_block_delta" in event_item:
			if _handle_converse_event(event_item):
				break
			continue

		# otherwise try invoke shape (it will no-op if no "chunk")
		if _handle_invoke_event(event_item):
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
		content = resp.get("output", {}).get("message", {}).get("content", [])
		parts: List[str] = []
		for item in content:
			if isinstance(item, dict):
				t = item.get("text")
				if isinstance(t, str):
					parts.append(t)
		return "".join(parts)
	except Exception:
		return ""

def build_converse_request(
		model_id: str,
		system_prompt: str,
		context: str,
		question: str,
		max_tokens: int,
		temperature: float,
		rag_instructions: str = "",
		**extra_params: Any,
) -> Dict[str, Any]:
	"""
	Build the shared request payload for converse() and converse_stream().
	"""
	user_text = f"Context:\n{context}\n\nQuestion:\n{question}\n"
	if rag_instructions:
		user_text = f"{rag_instructions}\n{user_text}"

	req: Dict[str, Any] = {
		"modelId": model_id,
		"system": [{"text": system_prompt}],
		"messages": [
			{
				"role": "user",
				"content": [{"text": user_text}],
			}
		],
		"inferenceConfig": {
			"maxTokens": max_tokens,
			"temperature": temperature,
		},
	}

	# Allow caller to pass model-specific / top-level extra fields.
	# (If a key collides, extra_params wins.)
	if extra_params:
		req.update(extra_params)

	return req

