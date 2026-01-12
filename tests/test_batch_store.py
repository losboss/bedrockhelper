import time
from unittest import TestCase
from datetime import datetime, timezone

from bedrockhelper import BatchJobResponse
from bedrockhelper.batch.store import InMemoryJobStore


def make_job(job_id: str, *, output_s3_uri: str = "s3://bucket/key") -> BatchJobResponse:
	return BatchJobResponse(
		job_id=job_id,
		job_name="name",
		model_id="model",
		input_s3_uri="s3://in/key",
		output_s3_uri=output_s3_uri,
		response={},  # optional
	)


class TestInMemoryJobStore(TestCase):

	def test_add_get_and_idempotent(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(make_job("job-1"))

		got = store.get("job-1")
		self.assertIsNotNone(got)
		self.assertEqual(got.job.job_id, "job-1")

		# Add again should be a no-op
		store.add(make_job("job-1"))
		all_unfinished = store.list_unfinished_jobs(limit=10)
		self.assertEqual(len(all_unfinished), 1)

	def test_upsert_new_and_preserve_fields_on_update(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(make_job("job-1"))

		# update some stored metadata via mark_checked (use the same id)
		now = datetime.now(timezone.utc)
		store.mark_checked(
			"job-1",
			state="RUNNING",
			attempts=2,
			last_error="err",
			bedrock_status="InProgress",
			last_checked_at=now
		)

		# Upsert with same id but changed payload; stored metadata should be preserved
		new_job_ref = make_job("job-1", output_s3_uri="s3://bucket/updated")
		store.upsert(new_job_ref)

		cur = store.get("job-1")
		self.assertIsNotNone(cur)
		self.assertEqual(cur.job.output_s3_uri, "s3://bucket/updated")
		self.assertEqual(cur.state, "RUNNING")
		self.assertEqual(cur.attempts, 2)
		self.assertEqual(cur.last_error, "err")
		self.assertEqual(cur.bedrock_status, "InProgress")
		self.assertEqual(cur.last_checked_at, now)

	def test_remove(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(make_job("job-3"))

		self.assertTrue(store.remove("job-3"))
		self.assertFalse(store.remove("job-3"))
		self.assertIsNone(store.get("job-3"))

	def test_list_unfinished_jobs_order_and_limit(self):
		store = InMemoryJobStore(max_jobs=10)
		j1 = make_job("job-old")
		j2 = make_job("job-new")
		store.add(j1)
		time.sleep(0.01)  # ensure order by created_at
		store.add(j2)

		first = store.list_unfinished_jobs(limit=1)
		self.assertEqual(len(first), 1)
		self.assertEqual(first[0].job.job_id, "job-old")

		both = store.list_unfinished_jobs(limit=10)
		self.assertEqual([s.job.job_id for s in both], ["job-old", "job-new"])

	def test_mark_processed_and_excluded_from_unfinished(self):
		store = InMemoryJobStore(max_jobs=10)
		job = make_job("job-4")
		store.add(job)

		processed_at = datetime.now(timezone.utc)
		store.mark_processed("job-4", processed_at=processed_at, embeddings_count=5)

		cur = store.get("job-4")
		self.assertIsNotNone(cur)
		self.assertEqual(cur.state, "PROCESSED")
		self.assertEqual(cur.embeddings_count, 5)
		self.assertIsNone(cur.last_error)

		unfinished = store.list_unfinished_jobs(limit=10)
		self.assertNotIn("job-4", [s.job.job_id for s in unfinished])

	def test_mark_checked_updates_fields(self):
		store = InMemoryJobStore(max_jobs=10)
		job = make_job("job-5")
		store.add(job)

		now = datetime.now(timezone.utc)
		store.mark_checked("job-5", state="FAILED", attempts=3, last_error="boom", bedrock_status="Failed",
		                   last_checked_at=now)

		cur = store.get("job-5")
		self.assertEqual(cur.state, "FAILED")
		self.assertEqual(cur.attempts, 3)
		self.assertEqual(cur.last_error, "boom")
		self.assertEqual(cur.bedrock_status, "Failed")
		self.assertEqual(cur.last_checked_at, now)

	def test_cleanup_removes_old_terminal_jobs(self):
		# short TTL for test determinism
		store = InMemoryJobStore(max_jobs=10, terminal_ttl_seconds=1.0)
		job = make_job("job-6")
		store.add(job)

		# mark processed with processed_at older than TTL
		old_ts = datetime.fromtimestamp(time.time() - 2.0, timezone.utc)
		store.mark_processed("job-6", processed_at=old_ts, embeddings_count=1)

		removed = store.cleanup()
		self.assertGreaterEqual(removed, 1)
		self.assertIsNone(store.get("job-6"))

	def test_eviction_respects_max_jobs(self):
		# small capacity ensures eviction occurs
		store = InMemoryJobStore(max_jobs=1)
		store.add(make_job("a"))
		store.add(make_job("b"))  # should evict one to maintain bound

		remaining = store.list_unfinished_jobs(limit=10)
		self.assertEqual(len(remaining), 1)
		self.assertIn(remaining[0].job.job_id, {"a", "b"})
