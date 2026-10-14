"""
Command line entry point for the Ising VQE / adiabatic experiments.

Each analysis is run as a subcommand followed by its keyword arguments, given as
"option=value" tokens::

    python Code/main.py analyze_AQC_fidelity "res=5" "T_range=[10, 40]"

That layout is what the Makefile uses, and it is kept as is; what changed is
that an unknown command, an unknown option or an unparsable value is now
reported instead of being silently ignored. A typo such as a missing "=" used to
drop the option and run the analysis with its full default workload - for
analyze_AQC_fidelity that is a 100 x 100 grid of adiabatic evolutions.
"""

import ast
import sys

from IsingModel import *
from Circuit import *
from probability_analysis import *
from adiabatic_QC import *
from AQC_vs_VQE import *
from parameters import *


# Recognised options per subcommand. The keys must match the keyword argument
# names of the corresponding analysis function.
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
    "g_initial": None,
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
    "backend": None,
    "E0": None,
    "seed": None
}

SUBCOMMAND_ARGS = {
    "analyze_AQC_fidelity": analyze_AQC_fidelity_args,
    "analyze_Ry_phase_vs_CX_depolarizing_error": analyze_Ry_phase_vs_CX_depolarizing_error_args,
    "compare_AQC_vs_VQE": compare_AQC_vs_VQE_args,
}

SUBCOMMANDS = {
    "analyze_AQC_fidelity": analyze_AQC_fidelity,
    "analyze_Ry_phase_vs_CX_depolarizing_error": analyze_Ry_phase_vs_CX_depolarizing_error,
    "compare_AQC_vs_VQE": compare_AQC_vs_VQE,
}

USAGE = """\
usage: python Code/main.py COMMAND ["option=value" ...]
       python Code/main.py --list
       python Code/main.py --help

Each command runs one analysis and takes its keyword arguments as
"option=value" tokens, for example

    python Code/main.py analyze_AQC_fidelity "res=5" "T_range=[10, 40]"

Values are parsed with ast.literal_eval, so numbers, strings, lists and tuples
all work. The special value BACKEND resolves to the FakeSherbrooke noise model
used for hardware-inspired simulations, e.g. "backend=BACKEND".

Commands:
"""


def usage_text() -> str:
    """
    Usage message listing every subcommand and the options it accepts.
    """
    lines = [USAGE]

    for name, spec in SUBCOMMAND_ARGS.items():
        lines.append(f"  {name}")
        lines.append(f"      options: {', '.join(spec)}")

    return "\n".join(lines)


def parse_options(spec: dict, arg: list, i: int):
    """
    Reads the "option=value" tokens that follow ``arg[i]``.

    Parameters:
        spec:  Mapping of the accepted option names for the command.
        arg:   Full argument list.
        i:     Index of the command itself.

    Returns:
        (kwargs, next_index, error). ``kwargs`` holds the options that were
        given; ``next_index`` is the first index after the options, and
        ``error`` is None on success.
    """
    kwargs = {k: v for k, v in spec.items() if v is not None}

    j = i + 1

    while j < len(arg) and "=" in arg[j]:

        key, _, value = arg[j].partition("=")

        if key not in spec:
            return None, j, f"unknown option {key!r} for command {arg[i]!r}"

        if value == "BACKEND":
            kwargs[key] = BACKEND

        else:
            try:
                kwargs[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return None, j, f"could not parse the value of {key!r}: {value!r}"

        j += 1

    return kwargs, j, None


def parse_commands(argv: list):
    """
    Parses the whole argument list without running anything.

    Parsing first and executing second matters: a usage error in the last
    argument must not leave the earlier commands half-run, and a mistyped
    analyze_AQC_fidelity would otherwise start its 100 x 100 default grid before
    the error is noticed.

    Returns:
        (commands, error) where commands is a list of (name, kwargs) pairs, or
        (None, error) as soon as something is wrong.
    """
    commands = []
    i = 0

    while i < len(argv):

        name = argv[i]

        if name not in SUBCOMMANDS:
            return None, f"unknown command {name!r}"

        kwargs, i, error = parse_options(SUBCOMMAND_ARGS[name], argv, i)

        if error is not None:
            return None, error

        commands.append((name, kwargs))

    return commands, None


def run_main(argv=None, stdout=None, stderr=None) -> int:
    """
    Runs the requested subcommands.

    Parameters:
        argv:   Argument list (defaults to sys.argv[1:]).
        stdout: Stream for regular output.
        stderr: Stream for diagnostics.

    Returns:
        The process exit code: 0 on success, 2 on a usage error.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    if argv and argv[0] in ("-h", "--help"):
        print(usage_text(), file=stdout)
        return 0

    if argv and argv[0] == "--list":
        print("\n".join(SUBCOMMAND_ARGS), file=stdout)
        return 0

    if not argv:
        # Previously this exited 0 without doing anything, which looks exactly
        # like a successful run.
        print(usage_text(), file=stderr)
        return 2

    commands, error = parse_commands(argv)

    if error is not None:
        print(f"error: {error}\n", file=stderr)
        print(usage_text(), file=stderr)
        return 2

    assert commands is not None

    for name, kwargs in commands:
        SUBCOMMANDS[name](**kwargs)

    return 0


if __name__ == "__main__":
    sys.exit(run_main())
