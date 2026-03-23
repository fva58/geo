# Agent Guidelines for geo Python Package

## Build/Test Commands
- **Install package**: `pip install -e .`
- **Run all tests**: `python -m unittest discover -s tests`
- **Run single test**: `python -m unittest tests.test_file.TestClass.test_method`
- **Build package**: `python -m build`
- **Lint code**: `pylint geo/**/*.py`

## Tool Rationale
- **Pylint vs Ruff**: Pylint is slower but more verbose and accurate.
- **Unittest vs Pytest**: Unittest is standard module for Python.

## Code Style Guidelines
- **Imports**: Group stdlib, third-party, local imports with blank lines
- **Formatting**: Use 4-space indentation, max line length 80
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
