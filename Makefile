PYTHON ?= python3

.PHONY: smoke test figures verify-results verify-public-tree check-figures verify

smoke:
	$(PYTHON) src/smoke_test.py

test:
	$(PYTHON) -m unittest discover -s tests -v

figures:
	$(PYTHON) scripts/generate_figures.py

verify-results:
	$(PYTHON) scripts/verify_results.py

verify-public-tree:
	$(PYTHON) scripts/verify_public_tree.py

check-figures:
	$(PYTHON) scripts/generate_figures.py --check

verify: smoke test figures verify-results verify-public-tree check-figures
