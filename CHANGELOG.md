# Changelog

## [1.0.0](https://github.com/losboss/bedrockhelper/compare/bedrockhelper-v0.1.0...bedrockhelper-v1.0.0) (2026-02-05)


### ⚠ BREAKING CHANGES

* add release-please and PyPI publishing

### Features

* add release-please and PyPI publishing ([7ca8f19](https://github.com/losboss/bedrockhelper/commit/7ca8f19ee48e3a4a239db8cab53d20145854c845))
* allow overriding input and output data configs for batch inference job creation, fixed tests ([43144ee](https://github.com/losboss/bedrockhelper/commit/43144eec84cafa4fde7e2f9b84daa4a951371ca5))


### Bug Fixes

* add perms to github codeql workflow ([cd9ea7d](https://github.com/losboss/bedrockhelper/commit/cd9ea7d965233c48f2fd1e99551109b9710763a7))
* fixed mypy errors, added additional precommit hooks for linting, formatting, etc ([df0fe88](https://github.com/losboss/bedrockhelper/commit/df0fe881809e7ffd10a991e75936b512dc20de7b))


### Dependencies

* bump the python-dependencies group across 1 directory with 3 updates ([2c79b5b](https://github.com/losboss/bedrockhelper/commit/2c79b5b21f1bfeb662836a5967db9936520be275))

## [0.1.0](https://github.com/losboss/BedrockHelper/commits/main) (2026)

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
