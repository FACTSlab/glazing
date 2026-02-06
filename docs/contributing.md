# Contributing to Glazing

We welcome contributions to Glazing! This guide will help you get started with development.

## Getting Started

### Prerequisites

- Python 3.13 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/glazing.git
cd glazing
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/factslab/glazing.git
```

### Development Setup

1. Create a virtual environment:

```bash
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install in development mode:

```bash
pip install -e ".[dev,docs]"
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

4. Initialize datasets for testing:

```bash
glazing init
```

## Development Workflow

### Creating a Branch

Always create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### Making Changes

1. Make your changes
2. Add or update tests as needed
3. Update documentation if required
4. Ensure all tests pass

### Code Style

We use `ruff` for linting and formatting:

```bash
# Check code style
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy --strict src/
```

The project enforces:
- Line length: 100 characters
- NumPy-style docstrings
- Type hints for all public APIs

### Running Tests

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_verbnet/test_models.py

# Run with coverage
pytest --cov=glazing --cov-report=html

# Run tests in parallel
pytest -n auto
```

### Testing Guidelines

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use fixtures for common test data
- Follow existing test patterns

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve documentation locally
mkdocs serve

# Build static documentation
mkdocs build
```

Documentation will be available at `http://127.0.0.1:8000/`

### Writing Documentation

- Update docstrings for any new or modified functions
- Use NumPy-style docstrings
- Include examples in docstrings when helpful
- Update user guides for significant features

## Submitting Changes

### Pre-submission Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy --strict src/`)
- [ ] Documentation builds without warnings (`mkdocs build`)
- [ ] Commit messages are clear and descriptive

### Creating a Pull Request

1. Push your changes:

```bash
git push origin feature/your-feature-name
```

2. Create a pull request on GitHub
3. Fill out the PR template
4. Link any related issues

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Write clear, descriptive commit messages
- Update CHANGELOG.md if adding features or fixing bugs
- Respond to review feedback promptly
- Ensure CI passes

## Project Structure

```
glazing/
├── src/glazing/          # Main package code
│   ├── framenet/         # FrameNet models and utilities
│   ├── propbank/         # PropBank models and utilities
│   ├── verbnet/          # VerbNet models and utilities
│   ├── wordnet/          # WordNet models and utilities
│   ├── references/       # Cross-reference resolution
│   ├── cli/              # Command-line interface
│   └── utils/            # Shared utilities
├── tests/                # Test suite
├── docs/                 # Documentation
├── pyproject.toml        # Project configuration
└── mkdocs.yml           # Documentation configuration
```

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` on GitHub. These are ideal for newcomers.

### Feature Requests

Check the issue tracker for `enhancement` labels. Feel free to discuss implementation approaches.

### Documentation

- Improve existing documentation
- Add more examples
- Fix typos or unclear explanations
- Translate documentation

### Performance

- Optimize data loading
- Improve search performance
- Reduce memory usage

### Testing

- Increase test coverage
- Add edge case tests
- Improve test fixtures

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for general questions
- Check existing issues before creating new ones

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

## Recognition

Contributors are recognized in:

- The project's [CHANGELOG.md](https://github.com/factslab/glazing/blob/main/CHANGELOG.md)
- GitHub's contributor graph
- Special mentions for significant contributions

## License

By contributing to Glazing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:

- Open an issue
- Start a discussion
- Contact the maintainers

Thank you for contributing to Glazing!
