# BedrockHelper

Python helpers for Amazon Bedrock: question answering over your own context
(RAG) through the Converse API, Titan text embeddings, and batch embedding jobs
with a small job tracker.

## Install

```sh
pip install bedrockhelper
```

Requires Python 3.11+ and AWS credentials allowed to call Bedrock (and S3, for
batch jobs). Credentials come from boto3's usual sources.

## Answer a question

```python
from bedrockhelper import BedrockHelper

helper = BedrockHelper(region_name="ca-central-1")

resp = helper.generate_with_rag(
    system_prompt="You are a helpful assistant.",
    context=["Doc 1 text", "Doc 2 text"],
    question="Summarize these documents.",
)
print(resp.text)
print(resp.metrics.input_tokens, resp.metrics.output_tokens)
```

The model defaults to `global.anthropic.claude-sonnet-5` and can be set with
`rag_model_id=` (or `AWS_MODEL_ID`), or per call with `model_id=`.

Bedrock errors such as access denied, throttling or a bad request are raised
as `botocore` `ClientError`s. Nothing retries them through another API.

## Stream the answer

```python
stream = helper.stream_with_rag(
    system_prompt="You are concise.",
    context="Some context",
    question="What does it say?",
)
for chunk in stream:
    print(chunk, end="")

# Token counts arrive at the end of the stream.
print(stream.result.metrics.output_tokens)
```

## Async

```python
resp = await helper.async_generate_with_rag(
    system_prompt="You are helpful.",
    context="Some context",
    question="What does it say?",
)
print(resp.text)

chunks, header_metrics, result = await helper.async_stream_with_rag(
    system_prompt="You are helpful.",
    context="Some context",
    question="What does it say?",
)
async for chunk in chunks:
    print(chunk, end="")
print((await result).metrics.output_tokens)
```

At most `max_concurrent_streams` (default 8) async streams run at once.

## Temperature

| You pass | What is sent |
|---|---|
| nothing | `0.1`, on models that accept a temperature |
| a `float` | your value |
| `None` | nothing; the model uses its own default |

Claude Opus 4.7 and later, Sonnet 5, Fable 5 and Mythos 5 reject sampling
parameters, so none are sent to them (an explicit `temperature` is dropped with
a warning). If another model rejects them, the request is retried once without
them.

## Claude-only invoke_model path

`prefer_converse=False` sends an Anthropic Messages body through
`invoke_model` / `invoke_model_with_response_stream` instead of Converse. Only
Claude models accept that body.

## Assuming a role

```python
helper = BedrockHelper(
    region_name="ca-central-1",
    role_arn="arn:aws:iam::123456789012:role/BedrockCaller",
)
```

`role_arn` defaults to the `AWS_ROLE_ARN` environment variable, read when the
helper is created; pass `role_arn=None` to assume no role even when it is set.
The assumed credentials refresh before they expire. If the process is already
running as that role (as on EKS, which also sets `AWS_ROLE_ARN`), it is not
assumed again.

Other settings you leave out are read from the environment in the same way:
`AWS_REGION` (default `ca-central-1`), `AWS_ROLE_SESSION_NAME`,
`AWS_EXTERNAL_ID`, `AWS_ASSUME_ROLE_DURATION` (seconds, default 3600),
`AWS_MODEL_ID` and `AWS_EMBEDDING_MODEL_ID`.

## Embeddings

Up to 3,000 records are embedded with direct calls, and `embed_texts` returns
the vectors:

```python
resp = helper.embed_texts({"a": "Hello world", "b": "Another document"})
print(len(resp.embeddings["a"]))
```

Above `batch_threshold` (3,000 by default), it submits a Bedrock batch job
through S3 and returns the job instead:

```python
job = helper.embed_texts(
    big_records,
    s3_bucket="my-bucket",
    role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
)

info = helper.wait_for_batch_job(job.job_id)
print(info["status"])

embeddings = helper.parse_batch_embeddings(output_s3_uri=job.output_s3_uri)
```

## Reconciling batch jobs

Instead of waiting, record each job and check on them from a scheduled worker:

```python
from bedrockhelper import InMemoryJobStore, reconcile_batch_embedding_jobs

store = InMemoryJobStore()
store.add(job)


def process_embeddings(job_ref, embeddings):
    print("Got", len(embeddings), "embeddings for", job_ref.job_id)


processed, failed, skipped = reconcile_batch_embedding_jobs(
    helper=helper,
    store=store,
    process_embeddings=process_embeddings,
)
```

`process_embeddings` runs again if it raised on an earlier pass, so make it
idempotent. `InMemoryJobStore` loses its jobs when the process exits. For
durable tracking, implement the `bedrockhelper.batch.JobStore` protocol over
your own database. `InMemoryJobStore` is the reference implementation.

## License

MIT
