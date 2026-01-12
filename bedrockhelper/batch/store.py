from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol

from bedrockhelper.batch.models import StoredBatchJob
from bedrockhelper.models import BatchJobResponse


class JobStore(Protocol):
	"""
	Storage contract for batch-job reconciliation.

	Implement this in your app for durable storage (Postgres/Redis/etc).
	The default store shipped by this library is in-memory.
	"""

	def add(self, job: BatchJobResponse) -> None:
		...

	def get(self, job_id: str) -> Optional[StoredBatchJob]:
		...

	def upsert(self, job: BatchJobResponse) -> None:
		"""
		Insert or replace an existing job record.

		Useful if you re-submit the same logical job, or if you want to
		refresh stored payload fields like output_s3_uri/job_name, etc.
		"""
		...

	def remove(self, job_id: str) -> bool:
		"""
		Remove a job by id.

		Returns True if a record was removed, False if it didn't exist.
		"""
		...

	def list_unfinished_jobs(self, *, limit: int) -> List[StoredBatchJob]:
		...

	def mark_checked(
			self,
			job_id: str,
			*,
			state: Optional[str] = None,
			last_checked_at: Optional[datetime] = None,
			attempts: Optional[int] = None,
			last_error: Optional[str] = None,
			bedrock_status: Optional[str] = None,
	) -> None:
		...

	def mark_processed(
			self,
			job_id: str,
			*,
			processed_at: datetime,
			embeddings_count: int,
	) -> None:
		...

	def cleanup(self) -> int:
		"""
		Optional maintenance hook.
		Returns number of records removed.
		"""
		...


class InMemoryJobStore(JobStore):
	"""
	Thread-safe, bounded, TTL-based in-memory store.

	"Production ready" for *long-running processes* where in-memory lifecycle
	is acceptable (workers, daemons, services). Not durable across restarts.

	Features:
	- thread-safe
	- bounded by max_jobs
	- TTL cleanup for terminal jobs
	- deterministic ordering (oldest first) for reconciliation
	"""

	def __init__(
			self,
			*,
			max_jobs: int = 10_000,
			terminal_ttl_seconds: float = 6 * 3600,  # keep terminal jobs for 6h by default
	) -> None:
		if max_jobs < 1:
			raise ValueError("max_jobs must be >= 1")
		if terminal_ttl_seconds <= 0:
			raise ValueError("terminal_ttl_seconds must be > 0")

		self._max_jobs = max_jobs
		self._terminal_ttl_seconds = terminal_ttl_seconds

		self._lock = threading.RLock()
		self._jobs: Dict[str, StoredBatchJob] = {}

	def add(self, job: BatchJobResponse) -> None:
		now = datetime.now(timezone.utc)
		job_ref = job.to_ref()

		with self._lock:
			if job_ref.job_id in self._jobs:
				return

			if len(self._jobs) >= self._max_jobs:
				self._evict_one_locked()

			self._jobs[job_ref.job_id] = StoredBatchJob(
				job=job_ref,
				state="SUBMITTED",
				created_at=now,
			)

	def get(self, job_id: str) -> Optional[StoredBatchJob]:
		with self._lock:
			return self._jobs.get(job_id)

	def upsert(self, job: BatchJobResponse) -> None:
		now = datetime.now(timezone.utc)
		job_ref = job.to_ref()

		with self._lock:
			cur = self._jobs.get(job_ref.job_id)
			if cur is None:
				if len(self._jobs) >= self._max_jobs:
					self._evict_one_locked()
				self._jobs[job_ref.job_id] = StoredBatchJob(
					job=job_ref,
					state="SUBMITTED",
					created_at=now,
				)
				return

			self._jobs[job_ref.job_id] = StoredBatchJob(
				job=job_ref,  # update the persisted reference
				state=cur.state,
				created_at=cur.created_at,
				last_checked_at=cur.last_checked_at,
				attempts=cur.attempts,
				last_error=cur.last_error,
				bedrock_status=cur.bedrock_status,
				processed_at=cur.processed_at,
				embeddings_count=cur.embeddings_count,
				debug=cur.debug,
			)

	def remove(self, job_id: str) -> bool:
		with self._lock:
			return self._jobs.pop(job_id, None) is not None

	def list_unfinished_jobs(self, *, limit: int) -> List[StoredBatchJob]:
		if limit < 1:
			return []
		with self._lock:
			# Oldest first gives fair progress; prefer non-terminal states
			unfinished = [
				j for j in self._jobs.values()
				if j.state not in ("FAILED", "PROCESSED")
			]
			unfinished.sort(key=lambda j: j.created_at)
			return unfinished[:limit]

	def mark_checked(
			self,
			job_id: str,
			*,
			state: Optional[str] = None,
			last_checked_at: Optional[datetime] = None,
			attempts: Optional[int] = None,
			last_error: Optional[str] = None,
			bedrock_status: Optional[str] = None,
	) -> None:
		with self._lock:
			cur = self._jobs.get(job_id)
			if cur is None:
				return

			self._jobs[job_id] = StoredBatchJob(
				job=cur.job,
				state=state or cur.state,
				created_at=cur.created_at,
				last_checked_at=last_checked_at or cur.last_checked_at,
				attempts=attempts if attempts is not None else cur.attempts,
				last_error=last_error if last_error is not None else cur.last_error,
				bedrock_status=bedrock_status if bedrock_status is not None else cur.bedrock_status,
				processed_at=cur.processed_at,
				embeddings_count=cur.embeddings_count,
			)

	def mark_processed(
			self,
			job_id: str,
			*,
			processed_at: datetime,
			embeddings_count: int,
	) -> None:
		with self._lock:
			cur = self._jobs.get(job_id)
			if cur is None:
				return

			self._jobs[job_id] = StoredBatchJob(
				job=cur.job,
				state="PROCESSED",
				created_at=cur.created_at,
				last_checked_at=cur.last_checked_at,
				attempts=cur.attempts,
				last_error=None,
				bedrock_status=cur.bedrock_status,
				processed_at=processed_at,
				embeddings_count=embeddings_count,
			)

	def cleanup(self) -> int:
		"""
		Remove terminal jobs that have aged beyond TTL.
		"""
		now = time.time()
		removed = 0
		with self._lock:
			to_delete: List[str] = []
			for job_id, stored in self._jobs.items():
				if stored.state not in ("FAILED", "PROCESSED"):
					continue

				anchor = stored.processed_at or stored.last_checked_at or stored.created_at
				age = now - anchor.timestamp()
				if age >= self._terminal_ttl_seconds:
					to_delete.append(job_id)

			for job_id in to_delete:
				del self._jobs[job_id]
				removed += 1

		return removed

	# -------- internal

	def _evict_one_locked(self) -> None:
		"""
		Evict one record to maintain max_jobs bound.
		Prefer evicting the oldest terminal job; otherwise evict oldest overall.
		"""
		# Prefer terminal eviction
		terminal = [j for j in self._jobs.values() if j.state in ("FAILED", "PROCESSED")]
		if terminal:
			terminal.sort(key=lambda j: j.created_at)
			del self._jobs[terminal[0].job.job_id]
			return

		# Otherwise evict oldest overall
		all_jobs = list(self._jobs.values())
		all_jobs.sort(key=lambda j: j.created_at)
		del self._jobs[all_jobs[0].job.job_id]
