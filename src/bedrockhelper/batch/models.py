from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Defined beside BatchJobResponse, which builds it; re-exported here for existing imports.
from bedrockhelper.models import BatchJobRef

__all__ = ['BatchJobRef', 'StoredBatchJob']


@dataclass(frozen=True, slots=True)
class StoredBatchJob:
	"""
	A persisted representation of a submitted batch job.

	`job` is the minimal durable reference; store/reconciler owns `state`.
	"""

	job: BatchJobRef
	state: str  # "SUBMITTED" | "RUNNING" | "COMPLETED" | "FAILED" | "PROCESSED"
	created_at: datetime
	last_checked_at: datetime | None = None
	attempts: int = 0
	last_error: str | None = None
	bedrock_status: str | None = None
	processed_at: datetime | None = None
	embeddings_count: int | None = None

	debug: dict[str, Any] = field(default_factory=dict)
