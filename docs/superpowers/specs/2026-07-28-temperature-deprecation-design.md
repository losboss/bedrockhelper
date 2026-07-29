# Graceful handling of `temperature` deprecation

**Date:** 2026-07-28
**Status:** Approved

## Problem

Newer Anthropic models reject the sampling parameters that BedrockHelper sends on
every request.

- Bedrock's Converse API declares `inferenceConfig.temperature` as **optional**;
  when omitted, the model's own default applies.
  ([InferenceConfiguration](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html))
- `temperature`, `topP`, and `topK` are removed on Claude Opus 4.7, Opus 4.8,
  Opus 5, Fable 5, and Mythos 5 — any value returns a `ValidationException`.
- Claude Sonnet 5 rejects **non-default** sampling values with the same error.
- Claude Sonnet 4.5 and Haiku 4.5 accept `temperature` on its own, but reject
  `temperature` and `topP` together.

BedrockHelper hardcodes `temperature: float = 0.1` on `generate_with_rag` and
`stream_with_rag`, and writes it unconditionally into `inferenceConfig`
(`utils.build_converse_request`) and into the `invoke_model` Claude-message body.
Pointing the library at `claude-sonnet-5` therefore fails on every call.

The failure is also hard to diagnose: when Converse raises, `generate_with_rag`
logs at `debug` and falls back to `invoke_model`, which sends the same rejected
parameter and fails again. The caller sees the second error, not the cause.

## Design

### 1. Model capability registry (`utils.py`)

```python
MODELS_WITHOUT_SAMPLING_PARAMS: Tuple[str, ...] = (
    'claude-opus-4-7', 'claude-opus-4-8', 'claude-opus-5',
    'claude-sonnet-5', 'claude-fable-5', 'claude-mythos-5',
)

def supports_sampling_params(model_id: str) -> bool: ...
```

Matching is a case-insensitive substring test against the model ID, so it holds
for bare (`anthropic.claude-sonnet-5-…-v1:0`), cross-region (`us.`,`eu.`), and
global (`global.`) prefixes alike.

Sonnet 4.5 and Haiku 4.5 are deliberately **not** listed — they accept
`temperature` alone. Their `temperature` + `topP` conflict is covered by the
retry path below.

### 2. Send `temperature` only when requested *and* accepted

`build_converse_request` takes `temperature: Optional[float] = None` and writes
`inferenceConfig.temperature` only when the caller supplied a value **and**
`supports_sampling_params(model_id)` is true. When a caller explicitly sets a
temperature on an incompatible model, log a warning naming the model — never
drop it silently. The `invoke_model` Claude-message body follows the same rule.

### 3. Retry once without sampling parameters

Two pure helpers in `utils.py`:

- `is_sampling_param_error(exc) -> bool` — true for a `ValidationException`
  whose message names `temperature`, `top_p`, `topP`, or `topK`.
- `strip_sampling_params(request) -> Tuple[dict, bool]` — returns a copy with
  those keys removed from `inferenceConfig` and `additionalModelRequestFields`,
  plus whether anything changed.

A client-side helper wraps `_call_with_refresh` so `converse`, `converse_stream`,
and both `invoke_model` paths share one behavior: on a sampling-parameter
`ValidationException`, strip the parameters, log a warning, and retry **once**.
If the stripped retry also fails, that error propagates. No loop, no silent
success.

This keeps the library correct against models released after this version.

### 4. Public API: three temperature states

`temperature` keeps its `0.1` default via an `UNSET` sentinel, so three states
are distinguishable:

| Argument | Behaviour |
|---|---|
| `UNSET` (omitted) | `DEFAULT_TEMPERATURE` (`0.1`) on models that accept it |
| `float` | the caller's explicit value |
| `None` | omit entirely; the model applies its own default |

The sentinel exists to keep logging honest. With the default model now Sonnet 5,
a plain `0.1` default would warn on *every* call about dropping a value the
caller never chose. `resolve_temperature()` warns only when discarding an
explicit choice, and logs at debug when discarding the library default.

This keeps the release **backward compatible**: a caller on Sonnet 4.5 or
Haiku 4.5 who passes no temperature still gets `0.1`, exactly as before.

### 5. Default model change

The default `rag_model_id` moves from
`global.anthropic.claude-sonnet-4-5-20250929-v1:0` to
`global.anthropic.claude-sonnet-5`, verified present in-account alongside its
`us.` and bare variants.

Existing consumers pin `rag_model_id` explicitly, so this affects only new
callers.

### Versioning

Ships as a **minor** (`1.2.0`). No caller changes behaviour: explicit
temperatures are still honoured where supported, and omitted temperatures still
resolve to `0.1` on models that accept one.

Future model families that reject sampling parameters need **no release** — the
retry path handles them. Adding one to `MODELS_WITHOUT_SAMPLING_PARAMS` is a
patch-level optimisation that saves a round trip, never a correctness fix.

### 6. Adjacent fix

`main.py` logs the Converse failure at `log.debug` before falling back to
`invoke_model`. Raise it to `log.warning` including the exception message, so
the originating error is visible rather than masked by the fallback's failure.

## Testing

- Registry matching: prefixed and unprefixed IDs, affected and unaffected Claude
  models, non-Claude models (Titan, Nova).
- `build_converse_request`: omits when `None`, includes when supported, omits and
  warns when explicitly set on an incompatible model.
- `strip_sampling_params`: removes from both locations, reports no-change
  correctly, does not mutate the input.
- `is_sampling_param_error`: matches on parameter names, rejects unrelated
  `ValidationException`s and non-`ClientError` exceptions.
- End-to-end retry on the Converse and `converse_stream` paths, and propagation
  when the stripped retry still fails, using the existing `Mock` runtime pattern
  in `tests/test_main.py`.
