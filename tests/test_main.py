"""Tests for the command-line entry point."""

import inspect

import pytest

import main


# --- usage / help ------------------------------------------------------------

def test_help_returns_zero_and_lists_every_command(capsys):
    assert main.run_main(["--help"]) == 0

    out = capsys.readouterr().out

    for name in main.SUBCOMMANDS:
        assert name in out


def test_short_help_flag_is_supported(capsys):
    assert main.run_main(["-h"]) == 0

    assert "usage:" in capsys.readouterr().out


def test_list_prints_the_command_names(capsys):
    assert main.run_main(["--list"]) == 0

    assert capsys.readouterr().out.split() == list(main.SUBCOMMANDS)


def test_no_arguments_is_a_usage_error(capsys):
    """
    Regression: running main.py with no arguments exited 0 without doing
    anything, which looks exactly like a successful run.
    """
    assert main.run_main([]) == 2

    captured = capsys.readouterr()

    assert captured.out == ""
    assert "usage:" in captured.err


def test_usage_documents_the_backend_sentinel():
    assert "BACKEND" in main.usage_text()


# --- argument errors ---------------------------------------------------------

def test_unknown_command_is_reported(capsys):
    assert main.run_main(["frobnicate"]) == 2

    assert "unknown command 'frobnicate'" in capsys.readouterr().err


def test_unknown_option_is_reported(capsys):
    assert main.run_main(["compare_AQC_vs_VQE", "bogus=1"]) == 2

    assert "unknown option 'bogus'" in capsys.readouterr().err


@pytest.mark.parametrize("token", ["T=not-a-number", "T=[unclosed", "T=1..2"])
def test_unparsable_value_is_reported(capsys, token):
    assert main.run_main(["compare_AQC_vs_VQE", token]) == 2

    assert "could not parse the value of 'T'" in capsys.readouterr().err


def test_option_without_an_equals_sign_does_not_run_the_command(capsys, monkeypatch):
    """
    Regression: a token without "=" stopped the option scan and the command ran
    with all of its defaults - for analyze_AQC_fidelity that is a 100 x 100 grid
    of adiabatic evolutions, launched by a typo.
    """
    calls = []
    monkeypatch.setitem(main.SUBCOMMANDS, "analyze_AQC_fidelity", lambda **kwargs: calls.append(kwargs))

    assert main.run_main(["analyze_AQC_fidelity", "res"]) == 2
    assert calls == []
    assert "unknown command 'res'" in capsys.readouterr().err


def test_a_usage_error_prevents_earlier_commands_from_running(capsys, monkeypatch):
    """Nothing is executed until the whole command line has been validated."""
    calls = []
    monkeypatch.setitem(main.SUBCOMMANDS, "analyze_AQC_fidelity", lambda **kwargs: calls.append(kwargs))

    assert main.run_main(["analyze_AQC_fidelity", "res=2", "bogus_command"]) == 2
    assert calls == []
    assert "unknown command 'bogus_command'" in capsys.readouterr().err


def test_bad_option_on_a_later_command_prevents_the_earlier_one(monkeypatch, capsys):
    calls = []
    monkeypatch.setitem(main.SUBCOMMANDS, "compare_AQC_vs_VQE", lambda **kwargs: calls.append(kwargs))

    assert main.run_main(["compare_AQC_vs_VQE", "T=1", "analyze_AQC_fidelity", "nope=1"]) == 2
    assert calls == []
    assert "unknown option 'nope'" in capsys.readouterr().err


# --- option parsing ----------------------------------------------------------

def test_absent_options_are_not_passed_on():
    kwargs, index, error = main.parse_options(main.compare_AQC_vs_VQE_args, ["compare_AQC_vs_VQE"], 0)

    assert (kwargs, index, error) == ({}, 1, None)


def test_values_are_parsed_with_literal_eval():
    argv = ["analyze_AQC_fidelity", "res=5", "T_range=[10, 40]", "save_to='out'"]

    kwargs, index, error = main.parse_options(main.analyze_AQC_fidelity_args, argv, 0)

    assert error is None
    assert kwargs == {"res": 5, "T_range": [10, 40], "save_to": "out"}
    assert index == 4


def test_backend_sentinel_resolves_to_the_backend_object():
    argv = ["analyze_AQC_fidelity", "backend=BACKEND"]

    kwargs, index, error = main.parse_options(main.analyze_AQC_fidelity_args, argv, 0)

    assert error is None
    assert kwargs["backend"] is main.BACKEND
    assert index == 2


def test_parsing_stops_at_the_next_command():
    argv = ["analyze_AQC_fidelity", "res=5", "compare_AQC_vs_VQE", "T=3"]

    kwargs, index, error = main.parse_options(main.analyze_AQC_fidelity_args, argv, 0)

    assert error is None
    assert kwargs == {"res": 5}
    assert index == 2


def test_unknown_option_returns_the_position_of_the_offending_token():
    argv = ["analyze_AQC_fidelity", "res=5", "nope=1"]

    kwargs, index, error = main.parse_options(main.analyze_AQC_fidelity_args, argv, 0)

    assert kwargs is None
    assert index == 2
    assert "nope" in error


# --- dispatch ----------------------------------------------------------------

def test_kwargs_reach_the_analysis_function(monkeypatch):
    seen = {}
    monkeypatch.setitem(main.SUBCOMMANDS, "compare_AQC_vs_VQE", lambda **kwargs: seen.update(kwargs))

    assert main.run_main(["compare_AQC_vs_VQE", "T=7", "n_qubits=3"]) == 0
    assert seen == {"T": 7, "n_qubits": 3}


def test_several_commands_run_in_order(monkeypatch):
    order = []
    monkeypatch.setitem(main.SUBCOMMANDS, "compare_AQC_vs_VQE", lambda **kwargs: order.append(("vqe", kwargs)))
    monkeypatch.setitem(main.SUBCOMMANDS, "analyze_AQC_fidelity", lambda **kwargs: order.append(("aqc", kwargs)))

    assert main.run_main(["compare_AQC_vs_VQE", "T=1", "analyze_AQC_fidelity", "res=2"]) == 0
    assert order == [("vqe", {"T": 1}), ("aqc", {"res": 2})]


def test_run_main_accepts_explicit_streams():
    import io

    out = io.StringIO()

    assert main.run_main(["--list"], stdout=out) == 0
    assert out.getvalue().strip() == "\n".join(main.SUBCOMMANDS)


@pytest.mark.parametrize("name", sorted(main.SUBCOMMANDS))
def test_option_table_matches_the_analysis_signature(name):
    """The option table must not drift away from the function it configures."""
    parameters = set(inspect.signature(main.SUBCOMMANDS[name]).parameters)

    assert set(main.SUBCOMMAND_ARGS[name]) == parameters


def test_every_subcommand_is_documented():
    text = main.usage_text()

    for name, spec in main.SUBCOMMAND_ARGS.items():
        assert name in text

        for option in spec:
            assert option in text
