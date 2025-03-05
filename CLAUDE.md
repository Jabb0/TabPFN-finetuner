# Development Guide for Claude

## Build & Run Commands
- Create environment: `uv venv`
- Install: `uv pip install -e ".[dev]"`
- Run: `python -m finetuning.main`
- Test: `pytest tests/`
- Test single: `pytest tests/path/to/test.py::test_name`
- Lint: `ruff check .`
- Format: `black .`
- Type check: `mypy .`

## Code Style Guidelines
- **Imports**: Group standard library, then third-party, then local imports
- **Formatting**: Black formatter with 88 character line length
- **Types**: Use type annotations for all functions and methods
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Error handling**: Use explicit try/except blocks with specific exceptions
- **Documentation**: Docstrings for all public functions and classes

This is a PyTorch Lightning project for finetuning TabPFN models using uv as package manager.