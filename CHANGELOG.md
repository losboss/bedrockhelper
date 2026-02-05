# Changelog

## [1.0.0](https://github.com/losboss/bedrockhelper/commits/main) (2026-02-05)

### Features

* Initial release of BedrockHelper
* Core `BedrockHelper` class for Amazon Bedrock API interactions
* RAG (Retrieval Augmented Generation) support with knowledge base queries
* Embedding generation for text inputs
* Batch inference job management
	* Submit and monitor batch embedding jobs
	* `InMemoryJobStore` for tracking job state
	* `reconcile_batch_embedding_jobs` for batch job reconciliation
* Support for both streaming and standard response modes
* Configurable input/output token limits
* S3 integration for batch job input/output data

### Infrastructure

* AWS credential handling with role assumption support
* Region configuration via environment variables
* Type annotations with mypy strict mode
* Pre-commit hooks for linting (ruff) and formatting
* Test suite with pytest and 100% coverage target
* MIT License
