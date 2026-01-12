import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

from bedrockhelper import BatchJobResponse
from bedrockhelper.batch import StoredBatchJob


class TestStoredBatchJob(unittest.TestCase):
	def setUp(self) -> None:
		super().setUp()
		self.job_obj = BatchJobResponse(
			job_id="test-job-id",
			job_name="test-job-name",
			model_id="test-model-id",
			input_s3_uri="s3://input-bucket/input-key",
			output_s3_uri="s3://output-bucket/output-key",
		)

	def test_defaults_and_fields(self):
		now = datetime.now(timezone.utc)
		sb = StoredBatchJob(job=self.job_obj, state="SUBMITTED", created_at=now)

		self.assertIs(sb.job, self.job_obj)
		self.assertEqual(sb.state, "SUBMITTED")
		self.assertEqual(sb.created_at, now)

		self.assertIsNone(sb.last_checked_at)
		self.assertEqual(sb.attempts, 0)
		self.assertIsNone(sb.last_error)
		self.assertIsNone(sb.bedrock_status)
		self.assertIsNone(sb.processed_at)
		self.assertIsNone(sb.embeddings_count)

	def test_frozen_dataclass_prevents_attribute_assignment(self):
		sb = StoredBatchJob(job=self.job_obj, state="SUBMITTED", created_at=datetime.now(timezone.utc))
		with self.assertRaises(FrozenInstanceError):
			sb.state = "RUNNING"

	def test_slots_prevents_instance_dict(self):
		sb = StoredBatchJob(job=self.job_obj, state="SUBMITTED", created_at=datetime.now(timezone.utc))
		self.assertFalse(hasattr(sb, "__dict__"))

	def test_equality(self):
		now = datetime.now(timezone.utc)
		sb1 = StoredBatchJob(job=self.job_obj, state="SUBMITTED", created_at=now)
		sb2 = StoredBatchJob(job=self.job_obj, state="SUBMITTED", created_at=now)
		self.assertEqual(sb1, sb2)

	def test_not_hashable(self):
		now = datetime.now(timezone.utc)
		sb = StoredBatchJob(job=self.job_obj, state="SUBMITTED", created_at=now)

		with self.assertRaises(TypeError):
			hash(sb)
