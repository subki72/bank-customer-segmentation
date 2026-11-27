# Contributing Guide

Thank you for contributing to **Bank Customer Segmentation**.

## Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install runtime and development dependencies:
   ```bash
   python -m pip install -r requirements.txt
   python -m pip install pytest pytest-cov ruff flake8 pre-commit
   pre-commit install
   ```

## Coding Standards

- Follow PEP 8 with max line length `120` enforced by **Ruff** and **Flake8**.
- Add full type hints (`-> None`, `list[...]`, `dict[...]`) and Google-style docstrings to all public functions and classes.
- Use `logging.getLogger(__name__)` inside `src/` and `main.py` instead of bare `print()`.
- Keep all unit and integration tests **100% offline** using synthetic generators or local sample fixtures.
- Never commit secrets, credentials, or `.env` files.

## Verification Checklist Before Opening a Pull Request

```bash
make lint
make test
make health
make demo
```
