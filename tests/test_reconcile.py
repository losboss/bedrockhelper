import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bedrockhelper.batch.reconcile import (
	state_from_bedrock_status,
	reconcile_batch_embedding_jobs,
)


class TestReconcileBatchEmbeddingJobs(unittest.TestCase):
	def _make_stored(self, job_id="job1", output_s3_uri="s3://bucket/key", attempts=0):
		return SimpleNamespace(job=SimpleNamespace(job_id=job_id, output_s3_uri=output_s3_uri), attempts=attempts)

	def test_state_from_bedrock_status_mappings(self):
		self.assertEqual(state_from_bedrock_status("Completed"), "COMPLETED")
		self.assertEqual(state_from_bedrock_status("Failed"), "FAILED")
		self.assertEqual(state_from_bedrock_status("Stopped"), "FAILED")
		self.assertEqual(state_from_bedrock_status("Expired"), "FAILED")
		self.assertEqual(state_from_bedrock_status("InProgress"), "RUNNING")
		self.assertEqual(state_from_bedrock_status("Unknown"), "RUNNING")

	def test_reconcile_successful_job_processes_embeddings(self):
		helper = Mock()
		store = Mock()
		stored = self._make_stored()
		store.list_unfinished_jobs.return_value = [stored]

		# Patch BatchJobResponse used inside reconcile to control status checks
		with patch("bedrockhelper.batch.reconcile.BatchJobResponse") as BJR:
			BJR.status.return_value = "Completed"
			BJR.is_success.return_value = True
			BJR.is_failure.return_value = False

			helper.get_batch_job.return_value = {"fake": "info"}
			embeddings = {"r1": [0.1, 0.2]}
			helper.parse_batch_embeddings.return_value = embeddings

			process_embeddings = Mock()

			processed, failed, skipped = reconcile_batch_embedding_jobs(
				helper=helper, store=store, process_embeddings=process_embeddings, limit=5
			)

			self.assertEqual(processed, 1)
			self.assertEqual(failed, 0)
			self.assertEqual(skipped, 0)

			process_embeddings.assert_called_once_with(stored.job, embeddings)
			store.mark_processed.assert_called_once()
			store.mark_checked.assert_called()

	def test_reconcile_completed_no_embeddings_marks_failed(self):
		helper = Mock()
		store = Mock()
		stored = self._make_stored()
		store.list_unfinished_jobs.return_value = [stored]

		with patch("bedrockhelper.batch.reconcile.BatchJobResponse") as BJR:
			BJR.status.return_value = "Completed"
			BJR.is_success.return_value = True
			BJR.is_failure.return_value = False

			helper.get_batch_job.return_value = {"fake": "info"}
			helper.parse_batch_embeddings.return_value = {}  # no embeddings

			processed, failed, skipped = reconcile_batch_embedding_jobs(
				helper=helper, store=store, process_embeddings=Mock()
			)

			self.assertEqual(processed, 0)
			self.assertEqual(failed, 1)
			self.assertEqual(skipped, 0)

			store.mark_checked.assert_called()
			called_kwargs = store.mark_checked.call_args.kwargs
			self.assertIn("last_error", called_kwargs)
			self.assertIn("parsed 0 embeddings", called_kwargs["last_error"])

	def test_reconcile_failure_marks_failed_with_summary(self):
		helper = Mock()
		store = Mock()
		stored = self._make_stored()
		store.list_unfinished_jobs.return_value = [stored]

		with patch("bedrockhelper.batch.reconcile.BatchJobResponse") as BJR:
			BJR.status.return_value = "Failed"
			BJR.is_success.return_value = False
			BJR.is_failure.return_value = True
			BJR.summary.return_value = "failure-summary"

			helper.get_batch_job.return_value = {"fake": "info"}

			processed, failed, skipped = reconcile_batch_embedding_jobs(
				helper=helper, store=store, process_embeddings=Mock()
			)

			self.assertEqual(processed, 0)
			self.assertEqual(failed, 1)
			self.assertEqual(skipped, 0)

			store.mark_checked.assert_called()
			called_kwargs = store.mark_checked.call_args.kwargs
			self.assertIn("last_error", called_kwargs)
			self.assertEqual(called_kwargs["last_error"], "failure-summary")

	def test_reconcile_non_terminal_job_is_skipped(self):
		helper = Mock()
		store = Mock()
		stored = self._make_stored()
		store.list_unfinished_jobs.return_value = [stored]

		with patch("bedrockhelper.batch.reconcile.BatchJobResponse") as BJR:
			BJR.status.return_value = "InProgress"
			BJR.is_success.return_value = False
			BJR.is_failure.return_value = False

			helper.get_batch_job.return_value = {"fake": "info"}

			processed, failed, skipped = reconcile_batch_embedding_jobs(
				helper=helper, store=store, process_embeddings=Mock()
			)

			self.assertEqual(processed, 0)
			self.assertEqual(failed, 0)
			self.assertEqual(skipped, 1)

			store.mark_checked.assert_called()

	def test_reconcile_helper_raises_exception_marks_skipped(self):
		helper = Mock()
		store = Mock()
		stored = self._make_stored()
		store.list_unfinished_jobs.return_value = [stored]

		helper.get_batch_job.side_effect = Exception("boom")

		processed, failed, skipped = reconcile_batch_embedding_jobs(
			helper=helper, store=store, process_embeddings=Mock()
		)

		self.assertEqual(processed, 0)
		self.assertEqual(failed, 0)
		self.assertEqual(skipped, 1)

		store.mark_checked.assert_called()
		called_kwargs = store.mark_checked.call_args.kwargs
		self.assertIn("last_error", called_kwargs)
		self.assertIn("boom", called_kwargs["last_error"])
	

	def test_reconcile_bedrock_status_none_skips_job(self):
		helper = Mock()
		store = Mock()
		stored = self._make_stored()
		store.list_unfinished_jobs.return_value = [stored]

		info = {"fake": "info"}
		helper.get_batch_job.return_value = info

		process_embeddings = Mock()

		with patch("bedrockhelper.batch.reconcile.BatchJobResponse") as BJR:
			BJR.status.return_value = None
			BJR.is_success.return_value = False
			BJR.is_failure.return_value = False

			processed, failed, skipped = reconcile_batch_embedding_jobs(
				helper=helper, store=store, process_embeddings=process_embeddings
			)

			self.assertEqual(processed, 0)
			self.assertEqual(failed, 0)
			self.assertEqual(skipped, 1)

			# Ensure heartbeat was recorded with a None bedrock_status
			store.mark_checked.assert_called()
			called_kwargs = store.mark_checked.call_args.kwargs
			self.assertIn("bedrock_status", called_kwargs)
			self.assertIsNone(called_kwargs["bedrock_status"])

			# Ensure no processing or finalization happened
			process_embeddings.assert_not_called()
			store.mark_processed.assert_not_called()
