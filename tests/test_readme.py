"""
Runs every ```python block in README.md, in order, against fake AWS clients.

The fakes check each call's parameters against botocore's real service model,
so a README example passing an argument Bedrock would reject fails here too.
"""

import ast
import asyncio
import contextlib
import inspect
import io
import json
import os
import re
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import boto3
from botocore.session import Session as BotocoreSession
from botocore.validate import validate_parameters
from stream_events import CONVERSE_EVENTS

from bedrockhelper.session import BotoSessionManager

README = Path(__file__).resolve().parent.parent / 'README.md'
PYTHON_BLOCK = re.compile(r'^```python\n(.*?)^```', re.MULTILINE | re.DOTALL)


class FakeClient:
	"""Answers like a boto3 client, validating each call against the real service model."""

	def __init__(self, service: str, responses: dict[str, Any]) -> None:
		# Only the bundled service model is needed. Explicit dummy keys stop boto3
		# looking up real credentials, which can reach the EC2 metadata endpoint.
		self._real = boto3.client(
			service,
			region_name='ca-central-1',
			aws_access_key_id='testing',
			aws_secret_access_key='testing',
		)
		self._responses = responses

	def __getattr__(self, name: str) -> Any:
		if name not in self._responses:
			raise AttributeError(name)
		api_name = self._real.meta.method_to_api_mapping.get(name)

		def call(*args: Any, **kwargs: Any) -> Any:
			if api_name is not None:
				validate_parameters(kwargs, self._real.meta.service_model.operation_model(api_name).input_shape)
			return self._responses[name](*args, **kwargs)

		return call


class FakePaginator:
	def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
		return [{'Contents': [{'Key': f'{Prefix}input.jsonl.out'}]}]


def batch_output(**_: Any) -> dict[str, Any]:
	lines = [json.dumps({'recordId': rid, 'modelOutput': {'embedding': [0.1, 0.2, 0.3]}}) for rid in ('doc-0', 'doc-1')]
	return {'Body': io.BytesIO('\n'.join(lines).encode())}


def fake_clients() -> dict[str, FakeClient]:
	return {
		'bedrock-runtime': FakeClient(
			'bedrock-runtime',
			{
				'converse': lambda **_: {
					'output': {'message': {'role': 'assistant', 'content': [{'text': 'Hello world'}]}},
					'usage': {'inputTokens': 12, 'outputTokens': 3, 'totalTokens': 15},
					'metrics': {'latencyMs': 321},
					'stopReason': 'end_turn',
				},
				'converse_stream': lambda **_: {'stream': iter(CONVERSE_EVENTS)},
				'invoke_model': lambda **_: {
					'body': io.BytesIO(json.dumps({'embedding': [0.1, 0.2, 0.3], 'inputTextTokenCount': 2}).encode())
				},
			},
		),
		'bedrock': FakeClient(
			'bedrock',
			{
				'create_model_invocation_job': lambda **_: {
					'jobArn': 'arn:aws:bedrock:ca-central-1:123456789012:model-invocation-job/abc123'
				},
				'get_model_invocation_job': lambda **_: {'status': 'Completed'},
			},
		),
		's3': FakeClient(
			's3',
			{
				'upload_file': lambda *args, **kwargs: None,
				'get_paginator': lambda name: FakePaginator(),
				'get_object': batch_output,
			},
		),
	}


def run_block(source: str, name: str, namespace: dict[str, Any]) -> None:
	code = compile(source, name, 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
	result = eval(code, namespace)  # noqa: S307 - our own README
	if inspect.iscoroutine(result):
		asyncio.run(result)


class TestReadmeExamples(TestCase):
	def test_every_python_block_runs(self):
		blocks = PYTHON_BLOCK.findall(README.read_text())
		self.assertGreaterEqual(len(blocks), 7, 'README python blocks not found')

		clients = fake_clients()
		# The README leaves `big_records` to the reader; it must exceed batch_threshold.
		namespace: dict[str, Any] = {'big_records': {f'doc-{i}': f'text {i}' for i in range(3001)}}
		out = io.StringIO()

		with patch.object(BotoSessionManager, 'client', lambda self, name: clients[name]):
			with contextlib.redirect_stdout(out):
				for n, block in enumerate(blocks, start=1):
					with self.subTest(block=n, first_line=block.splitlines()[0]):
						run_block(block, f'README.md python block {n}', namespace)

		printed = out.getvalue()
		self.assertIn('Hello world\n12 3', printed)  # non-streaming text and token counts
		self.assertIn('Hello world3', printed)  # streamed text, then the end-of-stream token count
		self.assertIn('Completed', printed)
		self.assertIn('Got 2 embeddings for arn:aws:bedrock', printed)

	def test_should_build_validation_clients_without_looking_up_credentials(self):
		# Credential lookup ends at the EC2 metadata endpoint, which an offline test must not reach.
		with patch.dict(os.environ, {}, clear=True):
			with patch.object(BotocoreSession, 'get_credentials', side_effect=AssertionError('credentials looked up')):
				fake_clients()
