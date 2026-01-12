from unittest import TestCase

from bedrockhelper.batch.models import BatchJobRef
from bedrockhelper.models import BatchJobResponse


class TestBatchJobResponse(TestCase):
	def test_status_non_string_returns_none(self):
		self.assertIsNone(BatchJobResponse.status({"status": 123}))
		self.assertIsNone(BatchJobResponse.status({}))  # missing

	def test_is_done_success_and_failure(self):
		self.assertTrue(BatchJobResponse.is_done({"status": "Completed"}))
		self.assertTrue(BatchJobResponse.is_done({"status": "Failed"}))
		self.assertFalse(BatchJobResponse.is_done({"status": "Running"}))

	def test_is_success_and_is_failure(self):
		self.assertTrue(BatchJobResponse.is_success({"status": "Completed"}))
		self.assertFalse(BatchJobResponse.is_success({"status": "Failed"}))

		self.assertTrue(BatchJobResponse.is_failure({"status": "Failed"}))
		self.assertFalse(BatchJobResponse.is_failure({"status": "Completed"}))

	def test_failure_reason_various_shapes(self):
		self.assertEqual(
			BatchJobResponse.failure_reason({"failureMessage": "oops"}), "oops"
		)
		self.assertEqual(
			BatchJobResponse.failure_reason({"message": "boom"}), "boom"
		)
		self.assertEqual(
			BatchJobResponse.failure_reason({"errorMessage": "err"}), "err"
		)
		self.assertEqual(
			BatchJobResponse.failure_reason({"error": {"message": "inner"}}), "inner"
		)
		self.assertEqual(
			BatchJobResponse.failure_reason({"failureDetails": {"message": "detail"}}),
			"detail",
		)
		# empty or non-string values yield None
		self.assertIsNone(BatchJobResponse.failure_reason({"message": ""}))
		self.assertIsNone(BatchJobResponse.failure_reason({"error": {"message": 123}}))

	def test_summary_formats_with_and_without_reason(self):
		self.assertEqual(
			BatchJobResponse.summary({"status": "Failed", "message": "boom"}),
			"Failed: boom",
		)
		self.assertEqual(
			BatchJobResponse.summary({"status": None}),
			"Unknown",
		)


class TestBatchJobRefConversion(TestCase):
	def test_to_ref_returns_equivalent_batch_job_ref(self):
		resp = BatchJobResponse(
			job_id="job-123",
			job_name="job-name",
			model_id="my-model",
			input_s3_uri="s3://input",
			output_s3_uri="s3://output",
			response={"status": "Completed"},
		)

		ref = resp.to_ref()

		self.assertIsInstance(ref, BatchJobRef)
		self.assertEqual(ref.job_id, "job-123")
		self.assertEqual(ref.job_name, "job-name")
		self.assertEqual(ref.model_id, "my-model")
		self.assertEqual(ref.input_s3_uri, "s3://input")
		self.assertEqual(ref.output_s3_uri, "s3://output")
