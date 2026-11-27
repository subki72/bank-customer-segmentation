## Summary

Describe the change, bug fix, or modeling update clearly.

## Type of Change

- [ ] Bug fix (non-breaking fix for an existing issue)
- [ ] New feature or endpoint
- [ ] Pipeline / modeling enhancement
- [ ] Documentation or CI/CD update

## Checklist

- [ ] Code passes `python -m ruff check src tests main.py` and `ruff format --check`
- [ ] Code passes `python -m flake8 --max-line-length=120 src tests main.py`
- [ ] Unit and integration tests pass (`python -m pytest tests/ -v --cov=src --cov-fail-under=75`)
- [ ] No hardcoded secrets, credentials, or untracked `.env` files introduced
- [ ] Relevant documentation (`README.md`, `CHANGELOG.md`, `docs/`) updated if needed
