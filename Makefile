PYTHON ?= python3

.PHONY: help reproduce test validate specs

help:
	@echo "Payload GNN-SHM — local helpers"
	@echo "  make reproduce  H3 spec validation + pytest (no Abaqus/GPU)"
	@echo "  make test       pytest only"
	@echo "  make validate   H3 fairing spec checks"
	@echo "  make specs      alias for validate"

reproduce:
	./scripts/reproduce_core.sh

test:
	$(PYTHON) -m pytest tests/ -q

validate specs:
	$(PYTHON) scripts/validate_h3_specs.py