from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

from .store import JobStore
from .models import BatchJobRef
from bedrockhelper.models import BatchJobResponse

ProcessEmbeddingsFn = Callable[[BatchJobResponse, Dict[str, List[float]]], None]


def state_from_bedrock_status(status: str) -> str:
	if status == 'Completed':
		return 'COMPLETED'
	if status in ('Failed', 'Stopped', 'Expired'):
		return 'FAILED'
	return 'RUNNING'


ProcessEmbeddingsFn = Callable[[BatchJobRef, Dict[str, List[float]]], None]


def reconcile_batch_embedding_jobs(
	*,
	helper: 'BedrockHelper',
	store: JobStore,
	process_embeddings: ProcessEmbeddingsFn,
	limit: int = 25,
) -> Tuple[int, int, int]:
	"""
	Poll and reconcile unfinished batch embedding jobs.

	Returns (processed, failed, skipped).
	"""
	now = datetime.now(timezone.utc)
	processed = 0
	failed = 0
	skipped = 0

	jobs = store.list_unfinished_jobs(limit=limit)
	for stored in jobs:
		job = stored.job
		attempts = stored.attempts + 1

		try:
			info = helper.get_batch_job(job.job_id)
			bedrock_status = BatchJobResponse.status(info)

			# Heartbeat
			store.mark_checked(
				job.job_id,
				last_checked_at=now,
				attempts=attempts,
				bedrock_status=bedrock_status,
			)

			if bedrock_status is None:
				skipped += 1
				continue

			if BatchJobResponse.is_success(info):
				embeddings = helper.parse_batch_embeddings(output_s3_uri=job.output_s3_uri)
				if not embeddings:
					store.mark_checked(
						job.job_id,
						state=state_from_bedrock_status(bedrock_status),
						last_checked_at=now,
						attempts=attempts,
						last_error='Completed but parsed 0 embeddings from output',
						bedrock_status=bedrock_status,
					)
					failed += 1
					continue

				# Caller-defined: upsert to pgvector, etc.
				process_embeddings(job, embeddings)

				store.mark_processed(
					job.job_id,
					processed_at=now,
					embeddings_count=len(embeddings),
				)
				processed += 1
				continue

			if BatchJobResponse.is_failure(info):
				store.mark_checked(
					job.job_id,
					state=state_from_bedrock_status(bedrock_status),
					last_checked_at=now,
					attempts=attempts,
					last_error=BatchJobResponse.summary(info),
					bedrock_status=bedrock_status,
				)
				failed += 1
				continue

			# Non-terminal: update state based on Bedrock status
			store.mark_checked(
				job.job_id,
				state=state_from_bedrock_status(bedrock_status),
				last_checked_at=now,
				attempts=attempts,
				bedrock_status=bedrock_status,
			)
			skipped += 1

		except Exception as e:
			store.mark_checked(
				job.job_id,
				last_checked_at=now,
				attempts=attempts,
				last_error=str(e),
			)
			skipped += 1

	return processed, failed, skipped
