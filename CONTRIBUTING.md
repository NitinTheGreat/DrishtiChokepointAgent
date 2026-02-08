# Contributing to DrishtiChokepointAgent

Thank you for your interest in contributing to DrishtiChokepointAgent!

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)
4. Install dev dependencies: `pip install -e ".[dev]"`

## Code Standards

### Style
- Python 3.11+ type hints required
- Run `ruff check .` before committing
- Run `mypy src/` for type checking
- Line length: 88 characters

### Documentation
- All public functions require docstrings
- Docstrings explain **why**, not just **what**
- Reference physics formulas in docstrings

### Testing
- Run tests: `pytest tests/ -v`
- New features require tests
- Aim for deterministic, reproducible tests

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Submit PR with clear description

## Design Principles

- **Determinism over performance**: Reproducible results first
- **Correctness over cleverness**: Simple, auditable code
- **Explicit over implicit**: No magic, no hidden state
- **Physics-grounded**: All metrics derive from crowd dynamics literature

## Questions?

Open an issue for discussion before major changes.
