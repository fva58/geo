# Agent Guidelines for geo Python Package

## Build/Test Commands
- **Install package**: `pip install -e .`
- **Run all tests**: `pytest tests/` (once tests are created) or `python -m unittest discover -s tests`
- **Run single test**: `pytest tests/test_file.py::test_function` or `python -m unittest tests.test_file.TestClass.test_method`
- **Build package**: `python -m build`
- **Lint code**: `ruff check .` (install ruff first)
- **Format code**: `ruff format .`

## Tool Rationale
- **Ruff vs Pylint**: Ruff is 10-100x faster, written in Rust, combines linting+formatting, and has excellent compatibility with Black. Pylint is slower and more verbose.
- **Pytest vs Unittest**: Pytest can run unittest tests, but write tests using unittest syntax for compatibility. Pytest provides better error messages and extensive plugin ecosystem.

## Code Style Guidelines
- **Imports**: Group stdlib, third-party, local imports with blank lines
- **Formatting**: Use 4-space indentation, max line length 88 (Black/Ruff default)
- **Types**: Add type hints for function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use specific exceptions, avoid bare except clauses
- **Documentation**: English language for comments and docstrings (per TODO.org)
- **Testing**: Write unit tests in `tests/` directory using unittest syntax (pytest can run them)
- **Packaging**: Follow setuptools configuration in pyproject.toml

## Project Context
- This is a geometry package for Riemannian spaces
- Target: PyPI package distribution
- Early stage project - tests directory currently empty
- MIT licensed, Russian author with English documentation requirement