PYTHON ?= python

.PHONY: check test lint docs package drift-check contract-compare clean

check: lint test docs drift-check

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m mkdocs build --strict

package:
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
