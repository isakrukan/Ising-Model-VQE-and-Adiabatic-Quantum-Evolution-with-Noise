# GNU Make convenience commands; a virtual environment can also be set up
# manually using the commands in README.md.

ifeq ($(OS),Windows_NT)
PYTHON ?= py -3
VENV_PYTHON := .venv/Scripts/python.exe

else
PYTHON ?= python3
VENV_PYTHON := .venv/bin/python

endif
.DEFAULT_GOAL := help
.PHONY: help setup dev run test lint clean


ARGS1 = analyze_AQC_fidelity "res=5" "T_range=[10,40]" "steps_range=[100,600]" "g=[0.01, 1]"

help:
	@echo "Targets:"
	@echo "  make setup  Create .venv and install dependencies"
	@echo "  make dev    Install the test and lint dependencies into .venv"
	@echo "  make run    Runs main.py with default command-line arguments"
	@echo "  make test   Run unit tests"
	@echo "  make lint   Run ruff"
	@echo "  make clean  Delete .venv"

setup:
	$(PYTHON) -m venv .venv
	"$(VENV_PYTHON)" -m pip install -r requirements.txt

dev:
	"$(VENV_PYTHON)" -m pip install -r requirements-dev.txt

run:
	"$(VENV_PYTHON)" Code/main.py $(ARGS1)
	"$(VENV_PYTHON)" Code/main.py $(ARGS1) "backend=BACKEND" "n_shots=30"

test:
	"$(VENV_PYTHON)" -m pytest

lint:
	"$(VENV_PYTHON)" -m ruff check .

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
