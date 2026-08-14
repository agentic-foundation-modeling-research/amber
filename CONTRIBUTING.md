# Contributing to Context Scythe

Thank you for your interest in contributing.

## Before you start

- Search existing issues and pull requests before opening a new one.
- Open an issue before making a large or behavior-changing contribution.
- Do not include credentials, private data, proprietary datasets, or internal service addresses in issues, commits, or test fixtures.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Context Scythe requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync --all-packages --extra dev
uv run playwright install chromium
```

Some integration and live tests require separately deployed WebArena services or model endpoints. Unit tests must not depend on those services.

## Testing

Run the test suite before submitting a pull request:

```sh
uv run pytest
```

If your change only affects one workspace package, you can run its tests directly:

```sh
uv run pytest packages/core
uv run pytest packages/env_server
```

Include tests for bug fixes and new behavior. If a test cannot be added, explain why in the pull request.

## Pull requests

Keep pull requests focused and include:

- a concise description of the problem and solution
- links to related issues
- the commands used to test the change
- documentation updates for user-visible behavior
- migration or compatibility notes when applicable

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
