from unittest import TestCase

from bedrockhelper.batch.models import BatchJobRef
from bedrockhelper import BatchJobResponse
from bedrockhelper.models import InvocationMetrics, RAGResponse, RAGStream


class TestBatchJobResponse(TestCase):
	def test_status_non_string_returns_none(self):
		self.assertIsNone(BatchJobResponse.status({'status': 123}))
		self.assertIsNone(BatchJobResponse.status({}))  # missing

	def test_is_done_success_and_failure(self):
		self.assertTrue(BatchJobResponse.is_done({'status': 'Completed'}))
		self.assertTrue(BatchJobResponse.is_done({'status': 'Failed'}))
		self.assertFalse(BatchJobResponse.is_done({'status': 'Running'}))

	def test_is_success_and_is_failure(self):
		self.assertTrue(BatchJobResponse.is_success({'status': 'Completed'}))
		self.assertFalse(BatchJobResponse.is_success({'status': 'Failed'}))

		self.assertTrue(BatchJobResponse.is_failure({'status': 'Failed'}))
		self.assertFalse(BatchJobResponse.is_failure({'status': 'Completed'}))

	def test_failure_reason_various_shapes(self):
		self.assertEqual(BatchJobResponse.failure_reason({'failureMessage': 'oops'}), 'oops')
		self.assertEqual(BatchJobResponse.failure_reason({'message': 'boom'}), 'boom')
		self.assertEqual(BatchJobResponse.failure_reason({'errorMessage': 'err'}), 'err')
		self.assertEqual(BatchJobResponse.failure_reason({'error': {'message': 'inner'}}), 'inner')
		self.assertEqual(
			BatchJobResponse.failure_reason({'failureDetails': {'message': 'detail'}}),
			'detail',
		)
		# empty or non-string values yield None
		self.assertIsNone(BatchJobResponse.failure_reason({'message': ''}))
		self.assertIsNone(BatchJobResponse.failure_reason({'error': {'message': 123}}))

	def test_summary_formats_with_and_without_reason(self):
		self.assertEqual(
			BatchJobResponse.summary({'status': 'Failed', 'message': 'boom'}),
			'Failed: boom',
		)
		self.assertEqual(
			BatchJobResponse.summary({'status': None}),
			'Unknown',
		)


class TestBatchJobRefConversion(TestCase):
	def test_to_ref_returns_equivalent_batch_job_ref(self):
		resp = BatchJobResponse(
			job_id='job-123',
			job_name='job-name',
			model_id='my-model',
			input_s3_uri='s3://input',
			output_s3_uri='s3://output',
			response={'status': 'Completed'},
		)

		ref = resp.to_ref()

		self.assertIsInstance(ref, BatchJobRef)
		self.assertEqual(ref.job_id, 'job-123')
		self.assertEqual(ref.job_name, 'job-name')
		self.assertEqual(ref.model_id, 'my-model')
		self.assertEqual(ref.input_s3_uri, 's3://input')
		self.assertEqual(ref.output_s3_uri, 's3://output')


class TestRAGStream(TestCase):
	def _make_stream(self, chunks: list[str]) -> RAGStream:
		"""Helper to create a RAGStream with given chunks."""
		header_metrics = InvocationMetrics(input_tokens=10)

		def iterator_factory():
			yield from chunks

		def build_final(full_text: str, tail_body):
			return RAGResponse(
				text=full_text,
				stream=True,
				metrics=InvocationMetrics(input_tokens=10, output_tokens=len(full_text)),
				raw_response=tail_body,
			)

		return RAGStream(
			iterator_factory=iterator_factory,
			header_metrics=header_metrics,
			build_final=build_final,
		)

	def test_metrics_property_returns_header_metrics(self):
		stream = self._make_stream(['hello'])
		# Access metrics property before iteration (line 73)
		self.assertEqual(stream.metrics.input_tokens, 10)

	def test_double_iteration_raises_error(self):
		stream = self._make_stream(['hello', ' world'])
		# First iteration succeeds
		list(stream)
		# Second iteration raises (line 77)
		with self.assertRaises(RuntimeError) as ctx:
			list(stream)
		self.assertIn('only be iterated once', str(ctx.exception))

	def test_result_before_iteration_raises_error(self):
		stream = self._make_stream(['hello'])
		# Access result before iteration (line 91)
		with self.assertRaises(RuntimeError) as ctx:
			_ = stream.result
		self.assertIn('not finished yet', str(ctx.exception))

	def test_set_tail_body(self):
		stream = self._make_stream(['hello'])
		# Set tail body before iteration
		stream._set_tail_body({'usage': {'tokens': 5}})
		# Iterate to completion
		list(stream)
		# Result should have the tail body
		self.assertEqual(stream.result.raw_response, {'usage': {'tokens': 5}})
