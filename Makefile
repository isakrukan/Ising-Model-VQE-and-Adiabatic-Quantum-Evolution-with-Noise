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
.PHONY: help setup run test clean

help:
	@echo "Targets:"
	@echo "  make setup  Create .venv and install dependencies"
	@echo "  make run    Runs main.py"
# 	@echo "  make test   Run unit tests"
	@echo "  make clean  Delete .venv"
.venv/pyvenv.cfg:
	$(PYTHON) -m venv .venv
setup: .venv/pyvenv.cfg
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PYTHON)" -m pip install -r requirements.txt
	@echo "Setup complete. Activation is optional when using make."
run:
	"$(VENV_PYTHON)" Code/main.py $(ARGS)
test:
# 	"$(VENV_PYTHON)" -m unittest discover -s tests -v
clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
