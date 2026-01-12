# BedrockHelper

A small, batteries-included helper for Amazon Bedrock:

- **RAG Q&A** using Bedrock Runtime `converse` / `converse_stream` when available, with automatic fallback.
- **Embeddings** using **Titan embed text v2**:
	- Small batches inline
	- Large batches via **Bedrock batch inference** (S3 JSONL)
- Works in **sync** and **async** codebases.
- Includes a **pluggable JobStore** abstraction for tracking batch jobs (in-memory, Redis, Postgres, etc.).

---

## Installation

This module is intended to be vendored or installed directly from source.

Requirements:

- Python 3.10+
- boto3 / botocore
- AWS credentials configured for Bedrock + S3

---

## Quick Start

```python
from bedrockhelper.main import BedrockHelper

helper = BedrockHelper(
    region_name="ca-central-1",
    rag_model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    embedding_model_id="amazon.titan-embed-text-v2:0",
)
```

---

## RAG Usage

### Non-streaming

```python
resp = helper.generate_with_rag(
    system_prompt="You are a helpful assistant.",
    context=["Doc 1", "Doc 2"],
    question="Summarize this",
)

print(resp.text)
```

### Streaming

```python
resp = helper.generate_with_rag(
    system_prompt="You are concise.",
    context="Some context",
    question="Stream this",
    stream=True,
)

for chunk in resp.stream:
    print(chunk, end="")
```

---

## Async Usage

```python
resp = await helper.async_generate_with_rag(
    system_prompt="You are helpful",
    context="Context",
    question="Async answer?",
)

print(resp.text)
```

Streaming:

```python
async for chunk in await helper.async_stream_generate_with_rag(
    system_prompt="You are helpful",
    context="Context",
    question="Async stream",
):
    print(chunk, end="")
```

---

## Embeddings

### Inline embeddings

```python
records = {
    "a": "Hello world",
    "b": "Another document",
}

resp = helper.embed_texts(records)
print(resp.embeddings["a"])
```

### Batch embeddings

```python
job = helper.embed_texts(
    big_records,
    s3_bucket="my-bucket",
    s3_prefix="bedrock_batch",
    role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
)

print(job.job_id)
```

Wait for completion:

```python
info = helper.wait_for_batch_job(job.job_id)
print(info["status"])
```

Parse results:

```python
embeddings = helper.parse_batch_embeddings(job.output_s3_uri)
```

---

## Batch Job Reconciliation

The helper includes a small framework for tracking and reconciling batch jobs.

```python
from bedrockhelper.batch.store import InMemoryJobStore
from bedrockhelper.batch.reconcile import reconcile_batch_embedding_jobs

store = InMemoryJobStore()
store.add(job)

def process_embeddings(job, embeddings):
    print("Processed", len(embeddings))

reconcile_batch_embedding_jobs(
    helper=helper,
    store=store,
    process_embeddings=process_embeddings,
)
```

---

## Implementing Your Own JobStore

To persist batch jobs across restarts or multiple workers, implement the `JobStore` interface.

### Redis Example

Use Redis keys per job plus a sorted set for ordering.

Key ideas:

- Store jobs as JSON
- Use TTL for terminal jobs
- Use a sorted set by `created_at`

This is ideal for horizontally scaled workers.

### Postgres Example

Suggested schema:

```sql
CREATE TABLE bedrock_batch_jobs (
  job_id TEXT PRIMARY KEY,
  job_name TEXT NOT NULL,
  model_id TEXT NOT NULL,
  input_s3_uri TEXT NOT NULL,
  output_s3_uri TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  last_checked_at TIMESTAMPTZ,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  bedrock_status TEXT,
  processed_at TIMESTAMPTZ,
  embeddings_count INTEGER,
  debug JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

This approach is boring, reliable, and production-friendly.

---

## Recommended Patterns

- Submit batch jobs quickly and return job IDs
- Reconcile in a background worker or cron
- Make `process_embeddings()` idempotent
- Prefer Postgres if you need observability

---

## License

Internal / project-specific. Adapt as needed.
