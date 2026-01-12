from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass(frozen=True, slots=True)
class BatchJobRef:
	"""
	Minimal, durable information needed to reconcile a Bedrock batch job later.

	Intentionally excludes large/unstable provider payloads.
	"""

	job_id: str
	job_name: str
	model_id: str
	input_s3_uri: str
	output_s3_uri: str


@dataclass(frozen=True, slots=True)
class StoredBatchJob:
	"""
	A persisted representation of a submitted batch job.

	`job` is the minimal durable reference; store/reconciler owns `state`.
	"""

	job: BatchJobRef
	state: str  # "SUBMITTED" | "RUNNING" | "COMPLETED" | "FAILED" | "PROCESSED"
	created_at: datetime
	last_checked_at: Optional[datetime] = None
	attempts: int = 0
	last_error: Optional[str] = None
	bedrock_status: Optional[str] = None
	processed_at: Optional[datetime] = None
	embeddings_count: Optional[int] = None

	debug: Dict[str, Any] = field(default_factory=dict)
