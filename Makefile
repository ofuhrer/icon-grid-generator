.PHONY: check test lint docs package clean

check: lint test docs

lint:
	python -m ruff check .

test:
	python -m pytest -q

docs:
	python -m mkdocs build --strict

package:
	rm -rf dist build
	python -m build
	python -m twine check dist/*

clean:
	rm -rf dist build site .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
