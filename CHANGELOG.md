# Changelog

## [1.2.0](https://github.com/losboss/bedrockhelper/compare/bedrockhelper-v1.1.0...bedrockhelper-v1.2.0) (2026-07-29)


### Features

* handle temperature deprecation on newer Anthropic models ([#30](https://github.com/losboss/bedrockhelper/issues/30)) ([bdaa71d](https://github.com/losboss/bedrockhelper/commit/bdaa71d7f5fdd66fc2b0de9c6e7f12ada7831201))


### Dependencies

* bump pytest from 9.0.2 to 9.0.3 ([#26](https://github.com/losboss/bedrockhelper/issues/26)) ([9be47f9](https://github.com/losboss/bedrockhelper/commit/9be47f9ccef4a6b68a0bc2e0affd90eb51d6df7f))
* bump the python-dependencies group across 1 directory with 9 updates ([#27](https://github.com/losboss/bedrockhelper/issues/27)) ([6cb1607](https://github.com/losboss/bedrockhelper/commit/6cb1607b73597ef4988f78573d9a1358c1890c5c))
* bump urllib3 from 2.6.3 to 2.7.0 ([#29](https://github.com/losboss/bedrockhelper/issues/29)) ([8e93c73](https://github.com/losboss/bedrockhelper/commit/8e93c737037b3e193265d4158adcaf392701d6f1))

## [1.1.0](https://github.com/losboss/bedrockhelper/compare/bedrockhelper-v1.0.2...bedrockhelper-v1.1.0) (2026-02-05)


### Features

* allow overriding input and output data configs for batch inference job creation, fixed tests ([43144ee](https://github.com/losboss/bedrockhelper/commit/43144eec84cafa4fde7e2f9b84daa4a951371ca5))


### Bug Fixes

* add perms to github codeql workflow ([cd9ea7d](https://github.com/losboss/bedrockhelper/commit/cd9ea7d965233c48f2fd1e99551109b9710763a7))
* add release-please and PyPI publishing ([b5bf6e3](https://github.com/losboss/bedrockhelper/commit/b5bf6e3833d2ac093ca50f9360139bc5a0ee0130))
* fixed mypy errors, added additional precommit hooks for linting, formatting, etc ([df0fe88](https://github.com/losboss/bedrockhelper/commit/df0fe881809e7ffd10a991e75936b512dc20de7b))
* resolving issues with tag immutability ([3d7750a](https://github.com/losboss/bedrockhelper/commit/3d7750a5f6b1aceb086b575c6cb52f2db357e3ba))
* resolving issues with tag immutability, update CHANGELOG to have correct starting tag ([#16](https://github.com/losboss/bedrockhelper/issues/16)) ([f0b823a](https://github.com/losboss/bedrockhelper/commit/f0b823a62f4b3ea70a154739e7248775312115d6))


### Dependencies

* bump the python-dependencies group across 1 directory with 3 updates ([2c79b5b](https://github.com/losboss/bedrockhelper/commit/2c79b5b21f1bfeb662836a5967db9936520be275))

## [1.0.2](https://github.com/losboss/bedrockhelper/commits/main) (2026-02-05)

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
