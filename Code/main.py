import sys
import ast
from IsingModel import *
from Circuit import *
from probability_analysis import *
from adiabatic_QC import *
from AQC_vs_VQE import *
from parameters import *


def setup_args(args_dict: dict, arg: list, i: int):

    j = i + 1

    while j < len(arg) and "=" in arg[j]:

        key, value = arg[j].split("=", 1)

        if key in args_dict:
            if value == "BACKEND":
                args_dict[key] = BACKEND
            else:
                try:
                    args_dict[key] = ast.literal_eval(value)
                except SyntaxError: # Probably a "_" in a filename
                    args_dict[key] = str(value)

        j += 1

    return j - 1

analyze_AQC_fidelity_args = {
    "D": None,
    "n_qubits": None, 
    "res": None, 
    "T_range": None, 
    "steps_range": None, 
    "J": None, 
    "h": None, 
    "g": None, 
    "n_shots": None,
    "save_to": None,
    "backend": None
    }

analyze_Ry_phase_vs_CX_depolarizing_error_args = {
    "N": None,
    "p_ry": None,
    "p_cx": None,
    "D": None,
    "n_qubits": None,
    "J": None,
    "h": None,
    "g": None,
    "save_to": None
}

compare_AQC_vs_VQE_args = {
    "D": None,
    "n_qubits": None,
    "g": None,
    "steps": None,
    "T": None,
    "n_shots": None,
    "backend": None
}

find_best_values_with_noise_args = {
    "name": None
}

if __name__ == "__main__":

    arg = sys.argv[1:]
    i = 0

    while i < len(arg):

        if arg[i] == "analyze_AQC_fidelity":
            i = setup_args(analyze_AQC_fidelity_args, arg, i)

            kwargs = {k: v for k, v in analyze_AQC_fidelity_args.items() if v is not None}
            analyze_AQC_fidelity(**kwargs)

        elif arg[i] == "analyze_Ry_phase_vs_CX_depolarizing_error":
            i = setup_args(analyze_Ry_phase_vs_CX_depolarizing_error_args, arg, i)

            kwargs = {k: v for k, v in analyze_Ry_phase_vs_CX_depolarizing_error_args.items() if v is not None}
            analyze_Ry_phase_vs_CX_depolarizing_error(**kwargs)

        elif arg[i] == "compare_AQC_vs_VQE":
            i = setup_args(compare_AQC_vs_VQE_args, arg, i)

            kwargs = {k: v for k, v in compare_AQC_vs_VQE_args.items() if v is not None}
            compare_AQC_vs_VQE(**kwargs)

        elif arg[i] == "find_best_values_with_noise":
            i = setup_args(find_best_values_with_noise_args, arg, i)

            kwargs = {k: v for k, v in find_best_values_with_noise_args.items() if v is not None}
            print(kwargs)
            find_best_values_with_noise(**kwargs)

        i += 1

    

    
