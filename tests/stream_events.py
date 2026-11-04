"""Bedrock response-stream events, shaped and ordered as Bedrock sends them."""

import json


def _chunk(msg):
	return {'chunk': {'bytes': json.dumps(msg).encode('utf-8')}}


# The events Bedrock sends, in order, for a two-delta answer. Usage arrives after
# the text is complete: in `metadata` after `messageStop` (Converse), or on the
# final `message_stop` chunk (invoke_model_with_response_stream).
CONVERSE_EVENTS = [
	{'messageStart': {'role': 'assistant'}},
	{'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'Hello'}}},
	{'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': ' world'}}},
	{'contentBlockStop': {'contentBlockIndex': 0}},
	{'messageStop': {'stopReason': 'end_turn'}},
	{
		'metadata': {
			'usage': {'inputTokens': 12, 'outputTokens': 3, 'totalTokens': 15},
			'metrics': {'latencyMs': 321},
		}
	},
]

INVOKE_EVENTS = [
	_chunk({'type': 'message_start', 'message': {'role': 'assistant', 'usage': {'input_tokens': 12}}}),
	_chunk({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
	_chunk({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello'}}),
	_chunk({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': ' world'}}),
	_chunk({'type': 'content_block_stop', 'index': 0}),
	_chunk({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 3}}),
	_chunk(
		{
			'type': 'message_stop',
			'amazon-bedrock-invocationMetrics': {
				'inputTokenCount': 12,
				'outputTokenCount': 3,
				'invocationLatency': 321,
				'firstByteLatency': 45,
			},
		}
	),
]
