# Contributing to Cogito

Thank you for considering contributing to Cogito! This document outlines the process for contributing to the project.

## Development Workflow

### Setting Up the Development Environment

1. Fork and clone the repository
2. Set up your development environment:

```bash
make build-dev
```

This will:
- Create a virtual environment with Python 3.10
- Install all dependencies
- Install the package in development mode

### Code Style and Pre-commit Hooks

We use Black for code formatting and pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
make pre-commit-install

# Run black formatting check
make code-style-check

# Apply black formatting
make code-style-dirty
```

## Making Changes

### Development Process

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit using commitizen for conventional commits (recommended):
   ```bash
   cz commit
   ```

3. Run tests to ensure your changes don't break existing functionality:
   ```bash
   make run-test
   ```

## Pull Request Process

1. Ensure all tests pass locally:
   ```bash
   make run-test
   ```

2. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request from your fork to the original repository:
   - Go to the original repository's GitHub page (https://github.com/freepik-company/fc-py-cogito)
   - Click on the "Pull requests" tab
   - Click the "New pull request" button
   - Click on the "compare across forks" link
   - Select your fork from the "head repository" dropdown
   - Select your feature branch from the "compare" dropdown
   - Click "Create pull request"
   - Add a descriptive title and detailed description of your changes
   - Reference any related issues using the #issue-number syntax

4. Ensure the CI pipeline passes:
   - Tests run automatically on each PR
   - Code style checks are enforced

5. Request review from maintainers
6. Address any feedback or requested changes from reviewers
7. Once approved, your PR will be merged to the `main` branch

## Release Process

Releases are triggered by tag pushes:

1. Alpha/beta prereleases are used for testing
2. Final releases are created when a version tag is pushed

## Additional Information

- The project uses GitHub Actions for CI/CD
- PRs are automatically tested
- Tagged versions are automatically published to PyPI
- Pushes to `main` are published to TestPyPI 