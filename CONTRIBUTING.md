# Contributing to Prophetverse

First off, thank you for considering contributing to Prophetverse! It's people like you that make Prophetverse such a great tool.

If you are new to open source, we highly recommend checking out [Open Source Guides](https://opensource.guide/) to learn how to open issues, create pull requests, and more.

## Local Environment Setup

To contribute to the codebase, you will need to set up your local development environment. Prophetverse uses [Poetry](https://python-poetry.org/) for dependency management and [pre-commit](https://pre-commit.com/) to ensure code quality.

### Prerequisites

- Python 3.10 or newer (up to <3.14)
- [Poetry](https://python-poetry.org/docs/#installation)
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/felipeangelimvieira/prophetverse.git
   cd prophetverse
   ```

2. **Install dependencies:**
   Prophetverse uses `poetry` to manage dependencies. To install the project along with its development dependencies, run:
   ```bash
   poetry install --extras "dev"
   ```

3. **Install pre-commit hooks:**
   We use `pre-commit` to catch format issues, typing errors, and enforce coding standards. To install the hooks, run:
   ```bash
   poetry run pre-commit install
   ```
   *Note: We use `commitlint` (which requires Node/npm) among other hooks to enforce Conventional Commits. Ensure your commit messages follow this convention.*

4. **Running Tests:**
   You can run tests using `pytest` to ensure everything is working correctly:
   ```bash
   poetry run pytest
   ```

## Development Guidelines

- **Code Quality:** We use `black`, `isort`, `flake8`, and `mypy` for formatting, linting, and type checking. These are automatically verified via pre-commit hooks when you attempt to commit your code.
- **Commit Messages:** We follow Conventional Commits. Commits must be properly formatted (e.g., `feat: Add new feature`, `fix: Resolve issue #123`).

We look forward to your contributions!
