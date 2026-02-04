import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest import TestCase

from bedrockhelper import InMemoryJobStore


@dataclass(frozen=True, slots=True)
class DummyBatchJobResponse:
	"""
	Stand-in for BatchJobResponse that satisfies the store's runtime contract:
	- has .to_ref() returning an object with .job_id and other persisted fields
	"""

	job_id: str
	job_name: str = 'job-name'
	model_id: str = 'model-id'
	input_s3_uri: str = 's3://in/x'
	output_s3_uri: str = 's3://out/x'

	def to_ref(self) -> Any:
		# The store expects StoredBatchJob.job to be a "ref" object with these fields.
		return SimpleNamespace(
			job_id=self.job_id,
			job_name=self.job_name,
			model_id=self.model_id,
			input_s3_uri=self.input_s3_uri,
			output_s3_uri=self.output_s3_uri,
		)


class TestInMemoryJobStore(TestCase):
	# ---------------------------
	# __init__ branches
	# ---------------------------
	def test_init_rejects_invalid_max_jobs(self):
		with self.assertRaises(ValueError):
			InMemoryJobStore(max_jobs=0)

	def test_init_rejects_invalid_terminal_ttl(self):
		with self.assertRaises(ValueError):
			InMemoryJobStore(terminal_ttl_seconds=0)

	# ---------------------------
	# add() branches
	# ---------------------------
	def test_add_inserts_new_job(self):
		store = InMemoryJobStore(max_jobs=10)
		job = DummyBatchJobResponse('job-1')

		store.add(job)
		got = store.get('job-1')

		self.assertIsNotNone(got)
		self.assertEqual(got.job.job_id, 'job-1')
		self.assertEqual(got.state, 'SUBMITTED')
		self.assertIsNotNone(got.created_at)

	def test_add_idempotent_if_exists(self):
		store = InMemoryJobStore(max_jobs=10)
		job = DummyBatchJobResponse('job-1')

		store.add(job)
		first = store.get('job-1')
		self.assertIsNotNone(first)

		# adding again should be no-op (created_at should remain unchanged)
		store.add(job)
		second = store.get('job-1')
		self.assertIsNotNone(second)
		self.assertEqual(second.created_at, first.created_at)

	def test_add_evicts_when_at_capacity_prefers_terminal_job(self):
		# capacity 2; we will add A and B, then mark A terminal and ensure A is evicted when adding C
		store = InMemoryJobStore(max_jobs=2)
		a = DummyBatchJobResponse('a')
		b = DummyBatchJobResponse('b')
		c = DummyBatchJobResponse('c')

		store.add(a)
		time.sleep(0.01)  # ensure created_at ordering
		store.add(b)

		# make 'a' terminal (FAILED) so eviction prefers it
		store.mark_checked('a', state='FAILED', last_checked_at=datetime.now(timezone.utc))

		store.add(c)  # triggers eviction
		self.assertIsNone(store.get('a'))
		self.assertIsNotNone(store.get('b'))
		self.assertIsNotNone(store.get('c'))

	def test_add_evicts_when_at_capacity_oldest_overall_if_no_terminal(self):
		store = InMemoryJobStore(max_jobs=2)
		a = DummyBatchJobResponse('a')
		b = DummyBatchJobResponse('b')
		c = DummyBatchJobResponse('c')

		store.add(a)
		time.sleep(0.01)
		store.add(b)

		# none terminal -> oldest overall ('a') should be evicted
		store.add(c)
		self.assertIsNone(store.get('a'))
		self.assertIsNotNone(store.get('b'))
		self.assertIsNotNone(store.get('c'))

	# ---------------------------
	# get()
	# ---------------------------
	def test_get_missing_returns_none(self):
		store = InMemoryJobStore(max_jobs=10)
		self.assertIsNone(store.get('nope'))

	# ---------------------------
	# upsert() branches
	# ---------------------------
	def test_upsert_inserts_when_missing(self):
		store = InMemoryJobStore(max_jobs=10)
		job = DummyBatchJobResponse('job-2', output_s3_uri='s3://out/original')
		store.upsert(job)

		got = store.get('job-2')
		self.assertIsNotNone(got)
		self.assertEqual(got.job.output_s3_uri, 's3://out/original')
		self.assertEqual(got.state, 'SUBMITTED')

	def test_upsert_inserts_when_missing_and_evicts_if_full(self):
		store = InMemoryJobStore(max_jobs=1)
		store.add(DummyBatchJobResponse('a'))
		store.upsert(DummyBatchJobResponse('b'))  # must evict one

		remaining = [j.job.job_id for j in store.list_unfinished_jobs(limit=10)]
		self.assertEqual(len(remaining), 1)
		self.assertIn(remaining[0], {'a', 'b'})
		self.assertTrue(store.get('a') is None or store.get('b') is None)

	def test_upsert_updates_ref_but_preserves_metadata(self):
		store = InMemoryJobStore(max_jobs=10)
		job = DummyBatchJobResponse('job-3', output_s3_uri='s3://out/v1')
		store.add(job)

		checked_at = datetime.now(timezone.utc)
		store.mark_checked(
			'job-3',
			state='RUNNING',
			attempts=7,
			last_error='boom',
			bedrock_status='InProgress',
			last_checked_at=checked_at,
		)

		# upsert same id with updated output uri and name
		job2 = DummyBatchJobResponse('job-3', job_name='new-name', output_s3_uri='s3://out/v2')
		store.upsert(job2)

		cur = store.get('job-3')
		self.assertIsNotNone(cur)

		# updated ref fields
		self.assertEqual(cur.job.job_name, 'new-name')
		self.assertEqual(cur.job.output_s3_uri, 's3://out/v2')

		# preserved metadata
		self.assertEqual(cur.state, 'RUNNING')
		self.assertEqual(cur.attempts, 7)
		self.assertEqual(cur.last_error, 'boom')
		self.assertEqual(cur.bedrock_status, 'InProgress')
		self.assertEqual(cur.last_checked_at, checked_at)

	# ---------------------------
	# remove() branches
	# ---------------------------
	def test_remove_returns_true_when_present(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('job-4'))

		self.assertTrue(store.remove('job-4'))
		self.assertIsNone(store.get('job-4'))

	def test_remove_returns_false_when_missing(self):
		store = InMemoryJobStore(max_jobs=10)
		self.assertFalse(store.remove('nope'))

	# ---------------------------
	# list_unfinished_jobs() branches
	# ---------------------------
	def test_list_unfinished_jobs_limit_less_than_one(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('a'))
		self.assertEqual(store.list_unfinished_jobs(limit=0), [])

	def test_list_unfinished_jobs_filters_failed_and_processed(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('a'))
		store.add(DummyBatchJobResponse('b'))
		store.add(DummyBatchJobResponse('c'))

		store.mark_checked('b', state='FAILED', last_checked_at=datetime.now(timezone.utc))
		store.mark_processed('c', processed_at=datetime.now(timezone.utc), embeddings_count=1)

		unfinished = store.list_unfinished_jobs(limit=10)
		self.assertEqual([j.job.job_id for j in unfinished], ['a'])

	def test_list_unfinished_jobs_orders_oldest_first_and_applies_limit(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('old'))
		time.sleep(0.01)
		store.add(DummyBatchJobResponse('new'))

		one = store.list_unfinished_jobs(limit=1)
		self.assertEqual(len(one), 1)
		self.assertEqual(one[0].job.job_id, 'old')

		both = store.list_unfinished_jobs(limit=10)
		self.assertEqual([j.job.job_id for j in both], ['old', 'new'])

	# ---------------------------
	# mark_checked() branches
	# ---------------------------
	def test_mark_checked_noop_when_missing(self):
		store = InMemoryJobStore(max_jobs=10)
		# should not raise
		store.mark_checked('missing', state='RUNNING', attempts=1)

	def test_mark_checked_updates_all_fields_when_provided(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('job-5'))

		now = datetime.now(timezone.utc)
		store.mark_checked(
			'job-5',
			state='FAILED',
			last_checked_at=now,
			attempts=3,
			last_error='err',
			bedrock_status='Failed',
		)
		cur = store.get('job-5')
		self.assertIsNotNone(cur)
		self.assertEqual(cur.state, 'FAILED')
		self.assertEqual(cur.last_checked_at, now)
		self.assertEqual(cur.attempts, 3)
		self.assertEqual(cur.last_error, 'err')
		self.assertEqual(cur.bedrock_status, 'Failed')

	def test_mark_checked_preserves_fields_when_args_none(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('job-6'))

		t1 = datetime.now(timezone.utc)
		store.mark_checked(
			'job-6',
			state='RUNNING',
			last_checked_at=t1,
			attempts=2,
			last_error='x',
			bedrock_status='InProgress',
		)
		cur1 = store.get('job-6')
		self.assertIsNotNone(cur1)

		# provide only state; others should remain
		store.mark_checked('job-6', state='RUNNING')
		cur2 = store.get('job-6')
		self.assertIsNotNone(cur2)

		self.assertEqual(cur2.last_checked_at, t1)
		self.assertEqual(cur2.attempts, 2)
		self.assertEqual(cur2.last_error, 'x')
		self.assertEqual(cur2.bedrock_status, 'InProgress')

	# ---------------------------
	# mark_processed() branches
	# ---------------------------
	def test_mark_processed_noop_when_missing(self):
		store = InMemoryJobStore(max_jobs=10)
		store.mark_processed('missing', processed_at=datetime.now(timezone.utc), embeddings_count=1)

	def test_mark_processed_sets_processed_state_and_clears_error(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('job-7'))
		store.mark_checked('job-7', last_error='err', attempts=2)

		pa = datetime.now(timezone.utc)
		store.mark_processed('job-7', processed_at=pa, embeddings_count=5)

		cur = store.get('job-7')
		self.assertIsNotNone(cur)
		self.assertEqual(cur.state, 'PROCESSED')
		self.assertEqual(cur.processed_at, pa)
		self.assertEqual(cur.embeddings_count, 5)
		self.assertIsNone(cur.last_error)

	# ---------------------------
	# cleanup() branches
	# ---------------------------
	def test_cleanup_removes_only_terminal_jobs_older_than_ttl(self):
		store = InMemoryJobStore(max_jobs=10, terminal_ttl_seconds=10.0)
		store.add(DummyBatchJobResponse('keep-nonterminal'))
		store.add(DummyBatchJobResponse('keep-terminal-young'))
		store.add(DummyBatchJobResponse('drop-terminal-old'))

		now = datetime.now(timezone.utc)

		# terminal young: processed_at within TTL
		store.mark_processed('keep-terminal-young', processed_at=now, embeddings_count=1)

		# terminal old: processed_at older than TTL
		old = now - timedelta(seconds=20)
		store.mark_processed('drop-terminal-old', processed_at=old, embeddings_count=1)

		removed = store.cleanup()
		self.assertGreaterEqual(removed, 1)
		self.assertIsNone(store.get('drop-terminal-old'))
		self.assertIsNotNone(store.get('keep-terminal-young'))
		self.assertIsNotNone(store.get('keep-nonterminal'))

	def test_cleanup_anchor_falls_back_last_checked_then_created(self):
		store = InMemoryJobStore(max_jobs=10, terminal_ttl_seconds=1.0)
		store.add(DummyBatchJobResponse('failed-old-by-last_checked'))
		store.add(DummyBatchJobResponse('failed-old-by-created'))

		# failed with last_checked_at old
		old_ts = datetime.fromtimestamp(time.time() - 2.0, tz=timezone.utc)
		store.mark_checked('failed-old-by-last_checked', state='FAILED', last_checked_at=old_ts)

		# failed without last_checked_at -> anchor becomes created_at; force it old by hacking internal state
		# (safe for unit tests; we're testing anchor selection logic)
		cur = store.get('failed-old-by-created')
		self.assertIsNotNone(cur)
		store._jobs['failed-old-by-created'] = type(cur)(
			job=cur.job,
			state='FAILED',
			created_at=old_ts,
			last_checked_at=None,
			attempts=cur.attempts,
			last_error=cur.last_error,
			bedrock_status=cur.bedrock_status,
			processed_at=cur.processed_at,
			embeddings_count=cur.embeddings_count,
		)

		removed = store.cleanup()
		self.assertGreaterEqual(removed, 2)
		self.assertIsNone(store.get('failed-old-by-last_checked'))
		self.assertIsNone(store.get('failed-old-by-created'))

	# ---------------------------
	# _evict_one_locked() branches (indirect + direct)
	# ---------------------------
	def test_evict_one_locked_prefers_oldest_terminal(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('t1'))
		time.sleep(0.01)
		store.add(DummyBatchJobResponse('t2'))
		time.sleep(0.01)
		store.add(DummyBatchJobResponse('n1'))

		store.mark_checked('t1', state='FAILED', last_checked_at=datetime.now(timezone.utc))
		store.mark_processed('t2', processed_at=datetime.now(timezone.utc), embeddings_count=1)

		# direct call to cover branch
		with store._lock:
			store._evict_one_locked()

		# oldest terminal should be t1 (created earlier)
		self.assertIsNone(store.get('t1'))
		self.assertIsNotNone(store.get('t2'))
		self.assertIsNotNone(store.get('n1'))

	def test_evict_one_locked_falls_back_to_oldest_overall(self):
		store = InMemoryJobStore(max_jobs=10)
		store.add(DummyBatchJobResponse('oldest'))
		time.sleep(0.01)
		store.add(DummyBatchJobResponse('newer'))

		with store._lock:
			store._evict_one_locked()

		self.assertIsNone(store.get('oldest'))
		self.assertIsNotNone(store.get('newer'))
