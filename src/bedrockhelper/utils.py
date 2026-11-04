import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from bedrockhelper.types import RecordInput

log = logging.getLogger(__name__)

# Model families that reject sampling parameters (temperature / topP / topK).
# Anthropic removed them on Opus 4.7+, Fable 5 and Mythos 5; Sonnet 5 rejects
# any non-default value. Matched as substrings so bare, cross-region ("us.")
# and global ("global.") Bedrock model IDs all resolve correctly.
#
# Sonnet 4.5 and Haiku 4.5 are deliberately absent: they accept temperature on
# its own and only reject it alongside topP, which the retry path handles.
MODELS_WITHOUT_SAMPLING_PARAMS: tuple[str, ...] = (
	'claude-opus-4-7',
	'claude-opus-4-8',
	'claude-opus-5',
	'claude-sonnet-5',
	'claude-fable-5',
	'claude-mythos-5',
)

# Keys removed from a request when a model rejects sampling parameters.
SAMPLING_PARAM_KEYS: tuple[str, ...] = ('temperature', 'topP', 'top_p', 'topK', 'top_k')

# Applied when the caller does not specify a temperature, on models that accept
# one. Kept deliberately low: this library is aimed at RAG, where grounded,
# low-variance answers matter more than creative range. Models that reject
# sampling parameters never see it.
DEFAULT_TEMPERATURE: float = 0.1


class _Unset:
	"""Sentinel type distinguishing "caller said nothing" from an explicit value."""

	def __repr__(self) -> str:  # pragma: no cover - debugging aid
		return '<unset>'


# Default for temperature arguments. Three states are meaningful:
#   UNSET  -> apply DEFAULT_TEMPERATURE where the model accepts it
#   float  -> the caller's explicit choice
#   None   -> send nothing; let the model apply its own default
UNSET = _Unset()

TemperatureArg = float | None | _Unset


def normalize_headers(headers: Mapping[str | int, Any]) -> dict[str, Any]:
	return {str(k).lower(): v for k, v in headers.items()}


def format_context_passages(
	passages: Sequence[str],
	*,
	include_headers: bool = False,
	header_template: str = '--- passage {i} ---',
	separator: str = '\n\n',
) -> str:
	blocks: list[str] = []
	for i, p in enumerate(passages, start=1):
		text = str(p)
		if include_headers:
			blocks.append(f'{header_template.format(i=i)}\n{text}')
		else:
			blocks.append(text)
	return separator.join(blocks)


def truncate_by_chars(text: str, max_chars: int | None) -> str:
	if max_chars is None:
		return text
	if max_chars <= 0:
		return ''
	return text[:max_chars]


def normalize_records(records: RecordInput) -> list[tuple[str, str]]:
	if isinstance(records, dict):
		return [(str(k), v) for k, v in records.items()]

	out: list[tuple[str, str]] = []
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


def extract_converse_text(resp: Mapping[str, Any]) -> str:
	"""
	Extract final assistant text from a Bedrock converse() response.
	"""
	output = resp.get('output')
	message = output.get('message') if isinstance(output, Mapping) else None
	content = message.get('content') if isinstance(message, Mapping) else None
	if not isinstance(content, list):
		return ''
	return ''.join(item['text'] for item in content if isinstance(item, Mapping) and isinstance(item.get('text'), str))


def iter_stream_text(
	event_stream: Iterable[Mapping[str, Any]],
	*,
	on_metrics: Callable[[dict[str, Any]], None],
) -> Iterator[str]:
	"""
	Yield the text deltas of a Bedrock response stream, passing end-of-stream metrics to `on_metrics`.

	Reads the stream to the end: the metrics arrive after the text has finished.

	- converse_stream(): text is in `contentBlockDelta` events; token usage and
	  latency come in the final `metadata` event, after `messageStop`. That
	  event's body ({'usage': ..., 'metrics': ...}) is passed on.
	- invoke_model_with_response_stream() with an Anthropic Messages body: each
	  event is `{'chunk': {'bytes': b'<json>'}}`; text is in `content_block_delta`
	  messages, and Bedrock adds `amazon-bedrock-invocationMetrics` to the final
	  `message_stop` message. A dict holding that key is passed on.
	"""
	for event in event_stream:
		block_delta = event.get('contentBlockDelta')
		if isinstance(block_delta, Mapping):
			text = (block_delta.get('delta') or {}).get('text')
			if isinstance(text, str) and text:
				yield text
			continue

		metadata = event.get('metadata')
		if isinstance(metadata, Mapping):
			on_metrics(dict(metadata))
			continue

		chunk = event.get('chunk')
		if isinstance(chunk, Mapping) and isinstance(chunk.get('bytes'), bytes):
			msg = json.loads(chunk['bytes'])
			if not isinstance(msg, dict):
				continue
			delta = msg.get('delta')
			if isinstance(delta, Mapping):
				text = delta.get('text')
				if isinstance(text, str) and text:
					yield text
			invocation_metrics = msg.get('amazon-bedrock-invocationMetrics')
			if isinstance(invocation_metrics, Mapping):
				on_metrics({'amazon-bedrock-invocationMetrics': dict(invocation_metrics)})


def supports_sampling_params(model_id: str) -> bool:
	"""
	Whether a model accepts sampling parameters (temperature / topP / topK).

	Newer Anthropic models reject them outright, so sending one returns a
	ValidationException. See MODELS_WITHOUT_SAMPLING_PARAMS.
	"""
	lowered = (model_id or '').lower()
	return not any(family in lowered for family in MODELS_WITHOUT_SAMPLING_PARAMS)


def strip_sampling_params(request: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
	"""
	Return a copy of a request with sampling parameters removed, along with
	whether anything was actually removed.

	Handles both request shapes: the nested inferenceConfig of converse(), and
	the top-level keys of an invoke_model() Claude message body.

	The input is not mutated.
	"""
	req = {k: v for k, v in request.items() if k not in SAMPLING_PARAM_KEYS}
	removed = len(req) != len(request)

	for section in ('inferenceConfig', 'additionalModelRequestFields'):
		values = req.get(section)
		if not isinstance(values, Mapping):
			continue
		kept = {k: v for k, v in values.items() if k not in SAMPLING_PARAM_KEYS}
		if len(kept) != len(values):
			req[section] = kept
			removed = True

	return req, removed


def is_sampling_param_error(exc: BaseException) -> bool:
	"""
	Whether an exception is Bedrock rejecting a sampling parameter.

	Matches the ValidationException raised when a model no longer accepts
	temperature / topP / topK, or refuses more than one of them at once.
	"""
	response = getattr(exc, 'response', None)
	if isinstance(response, Mapping):
		error = response.get('Error')
		code = error.get('Code') if isinstance(error, Mapping) else None
		if code and 'validationexception' not in str(code).lower():
			return False

	message = str(exc).lower()
	if 'validationexception' not in message and not isinstance(response, Mapping):
		return False

	return any(key.lower() in message for key in SAMPLING_PARAM_KEYS)


def resolve_temperature(model_id: str, temperature: TemperatureArg = UNSET) -> float | None:
	"""
	Decide the temperature to send for a model, or None to omit it entirely.

	Dropping a value the caller chose is worth a warning; dropping this
	library's own default is not, or every call to a model like Sonnet 5 would
	warn about a value the caller never asked for.
	"""
	if isinstance(temperature, _Unset):
		value: float | None = DEFAULT_TEMPERATURE
		explicit = False
	else:
		value = temperature
		explicit = True

	if value is None or supports_sampling_params(model_id):
		return value

	if explicit:
		log.warning(
			'Model %s does not accept sampling parameters; ignoring temperature=%s.',
			model_id,
			value,
		)
	else:
		log.debug('Model %s does not accept sampling parameters; omitting the default temperature.', model_id)
	return None


def build_converse_request(
	model_id: str,
	system_prompt: str,
	context: str,
	question: str,
	max_tokens: int,
	temperature: TemperatureArg = UNSET,
	rag_instructions: str = '',
	**extra_params: Any,
) -> dict[str, Any]:
	"""
	Build the shared request payload for converse() and converse_stream().

	temperature is omitted entirely on models that reject sampling parameters,
	and when the caller passes None.
	"""

	user_text = f'Context:\n{context}\n'
	if question:
		user_text += f'\nQuestion:\n{question}\n'
	if rag_instructions:
		user_text = f'{rag_instructions}\n{user_text}'

	inference_config: dict[str, Any] = {'maxTokens': max_tokens}
	resolved = resolve_temperature(model_id, temperature)
	if resolved is not None:
		inference_config['temperature'] = resolved

	req: dict[str, Any] = {
		'modelId': model_id,
		'system': [{'text': system_prompt}],
		'messages': [
			{
				'role': 'user',
				'content': [{'text': user_text}],
			}
		],
		'inferenceConfig': inference_config,
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
