.PHONY: install lint format test health demo serve docker-build clean

install:
	python -m pip install -r requirements.txt

lint:
	python -m ruff check src tests main.py
	python -m ruff format --check src tests main.py
	python -m flake8 --max-line-length=120 src tests main.py

format:
	python -m ruff format src tests main.py

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=75

health:
	python main.py --health

demo:
	python main.py --demo

serve:
	python -m uvicorn src.customer_segmentation.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t bank-customer-segmentation:latest .

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'dist', 'build')]"
