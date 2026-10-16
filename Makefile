# GNU Make convenience commands; a virtual environment can also be set up
# manually using the commands in README.md.

ifeq ($(OS),Windows_NT)
PYTHON ?= py -3
VENV_PYTHON := .venv/Scripts/python.exe
ACTIVATE := source .venv/Scripts/activate

else
PYTHON ?= python3
VENV_PYTHON := .venv/bin/python
ACTIVATE := source .venv/bin/activate

endif
.DEFAULT_GOAL := help
.PHONY: help setup run test clean

# Roughly 19 mins:
ARGS1 = analyze_AQC_fidelity "res=20" "T_range=[1,1000]" "steps_range=[1,1000]" "g=[0.01, 3]"
# Roughly 19 mins:
ARGS2 = analyze_AQC_fidelity "res=20" "T_range=[1,100]" "steps_range=[1,100]" "g=[0.01, 3]"
# Roughly 1 hour without backend and 7.5 hours with backend:
ARGS_long = analyze_AQC_fidelity "res=100" "T_range=[1,100]" "steps_range=[1,100]" "g=[0.01, 3]"

help:
	@echo "Targets:"
	@echo "  make setup  Create .venv and install dependencies"
	@echo "  make run    Runs main.py with default command-line arguments"
# 	@echo "  make test   Run unit tests"
	@echo "  make clean  Delete .venv"
.PHONY: setup run clean run_long

setup:
	$(PYTHON) -m venv .venv
	"$(VENV_PYTHON)" -m pip install -r requirements.txt

activate:
	@echo "Run: source .venv/Scripts/activate"

run:
	"$(VENV_PYTHON)" Code/main.py $(ARGS1)
	"$(VENV_PYTHON)" Code/main.py $(ARGS1) "backend=BACKEND" "n_shots=30"
	"$(VENV_PYTHON)" Code/main.py $(ARGS2)
	"$(VENV_PYTHON)" Code/main.py $(ARGS2) "backend=BACKEND" "n_shots=30"

run_long:
	@echo "Expected run time: 1 hour without backend and 7.5 hours with backend"
	"$(VENV_PYTHON)" Code/main.py $(ARGS_long)
	"$(VENV_PYTHON)" Code/main.py $(ARGS_long) "backend=BACKEND" "n_shots=30"

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
