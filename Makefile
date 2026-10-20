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
.PHONY: help setup activate run run_fidelity run_noise_analysis run_long clean


## Fidelity analysis
# Roughly 19 mins:
ARGS1 = analyze_AQC_fidelity "res=20" "T_range=[1,1000]" "steps_range=[1,1000]" "g=[0.01, 3]"
# Roughly 19 mins:
ARGS2 = analyze_AQC_fidelity "res=20" "T_range=[1,100]" "steps_range=[1,100]" "g=[0.01, 3]"
# Roughly 1 hour without backend and 7.5 hours with backend:
ARGS_long = analyze_AQC_fidelity "res=100" "T_range=[1,100]" "steps_range=[1,100]" "g=[0.01, 3]"

## Probability plot
ARGS3 = analyze_Ry_phase_vs_CX_depolarizing_error "N=20" "p_ry=[0,0.02]" "p_cx=[0,0.02]"

## AQC vs VQE comparison
# ARGS4 = compare_AQC_vs_VQE "g=0.01" "T=15" "steps=10"
ARGS4 = compare_AQC_vs_VQE "g=0.01" "T=20" "steps=100"
ARGS5 = compare_AQC_vs_VQE "g=0.01" "T=50" "steps=250"
ARGS6 = compare_AQC_vs_VQE "g=0.01" "T=100" "steps=500"

## Best fidelity
ARGS7 = find_best_values_with_noise "name=fidelities_D2_N4_J1_h0_g[0.01, 3]_T1.0-100.0_steps1-100_res100"

run_compare:
	"$(VENV_PYTHON)" Code/main.py $(ARGS4)
	"$(VENV_PYTHON)" Code/main.py $(ARGS4) "backend=BACKEND" "n_shots=30"

	"$(VENV_PYTHON)" Code/main.py $(ARGS5)
	"$(VENV_PYTHON)" Code/main.py $(ARGS5) "backend=BACKEND" "n_shots=30"

	"$(VENV_PYTHON)" Code/main.py $(ARGS6)
	"$(VENV_PYTHON)" Code/main.py $(ARGS6) "backend=BACKEND" "n_shots=30"

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

run_fidelity:
	"$(VENV_PYTHON)" Code/main.py $(ARGS1)
	"$(VENV_PYTHON)" Code/main.py $(ARGS1) "backend=BACKEND" "n_shots=30"
	"$(VENV_PYTHON)" Code/main.py $(ARGS2)
	"$(VENV_PYTHON)" Code/main.py $(ARGS2) "backend=BACKEND" "n_shots=30"

run_noise_analysis:
	"$(VENV_PYTHON)" Code/main.py $(ARGS3) 

run_best_fidelity:
	"$(VENV_PYTHON)" Code/main.py $(ARGS7)

run: run_fidelity run_noise_analysis run_compare

run_long:
	@echo "Expected run time: 1 hour without backend and 7.5 hours with backend"
	"$(VENV_PYTHON)" Code/main.py $(ARGS_long)
	"$(VENV_PYTHON)" Code/main.py $(ARGS_long) "backend=BACKEND" "n_shots=30"

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
