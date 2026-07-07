PYTHON ?= python

.PHONY: check test perf-check lint docs docs-figures docs-figures-check package drift-check contract-compare clean

check: lint test docs-figures-check docs drift-check

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q -m "not performance"

perf-check:
	GRID_GENERATOR_PERF_TESTS=1 $(PYTHON) -m pytest -q -m performance

docs:
	$(PYTHON) -m mkdocs build --strict

docs-figures:
	$(PYTHON) scripts/generate_docs_figures.py

docs-figures-check:
	$(PYTHON) scripts/generate_docs_figures.py --check

package: docs-figures-check
	rm -rf dist build
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

drift-check:
	git diff --check
	test -z "$$(git ls-files 'dist/*' 'build/*' 'site/*' 'tmp/*')"

contract-compare:
	test -n "$(REF_EXE)"
	REF_EXE="$(REF_EXE)" PYTHONPATH=src $(PYTHON) tmp/comparison/runs/contract_check.py

clean:
	rm -rf dist build site .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
