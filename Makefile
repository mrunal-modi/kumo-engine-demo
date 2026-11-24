# Set up the environment for the three demo notebooks. Run `make` to list the targets.
# Written for the GNU Make 3.81 that ships with macOS.

SHELL  := /bin/bash
PYTHON ?= python3
VENV   := .venv
STAMP  := $(VENV)/.installed

.DEFAULT_GOAL := help
.PHONY: help install wheel clean distclean

help: ## List the targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

install: $(STAMP) ## Build the kumo-engine wheel if needed and create .venv (first run: several minutes)
	@echo "Ready. Open the notebooks in VS Code, select the .venv kernel and run them in order: 1, 2, 3."

$(STAMP): requirements.txt
	@PYTHON=$(PYTHON) scripts/build_kumo_wheel.sh
	@echo "==> Creating $(VENV)/ and installing requirements.txt"
	@test -x $(VENV)/bin/python || $(PYTHON) -m venv $(VENV)
	@$(VENV)/bin/python -m pip install --quiet --upgrade pip
	@$(VENV)/bin/python -m pip install -r requirements.txt
	@touch $@

wheel: ## Build only the kumo-engine wheel, into scratch/wheels/
	@PYTHON=$(PYTHON) scripts/build_kumo_wheel.sh

clean: ## Delete the saved model and Kumo's working files (keeps .venv and the wheel)
	rm -rf models scratch/kumo_work

distclean: clean ## Also delete .venv/ and all of scratch/, including the wheel and its build
	rm -rf $(VENV) scratch
