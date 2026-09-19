import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ipop import cli
from ipop.cli import build_parser, write_validation_report
from ipop.data import DatasetSummary
from ipop.metadata import MetadataJoinResult


def test_parser_exposes_six_commands() -> None:
    parser = build_parser()
    for command in ("download", "validate", "prepare-emission", "run", "report", "all"):
        args = parser.parse_args([command])
        assert args.command == command


def test_validation_report_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    write_validation_report(DatasetSummary(3952, 2238, 553, 16023), path)
    assert json.loads(path.read_text(encoding="utf-8"))["target_observations"] == 16023


@pytest.mark.parametrize("command", ("download", "validate", "prepare-emission", "run", "report"))
def test_main_dispatches_single_command(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    calls: list[str] = []
    for name, label in (
        ("command_download", "download"),
        ("command_validate", "validate"),
        ("command_prepare", "prepare-emission"),
        ("command_run", "run"),
        ("command_report", "report"),
    ):
        monkeypatch.setattr(cli, name, lambda label=label: calls.append(label))

    assert cli.main([command]) == 0
    assert calls == [command]


def test_main_all_runs_commands_in_exact_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    for name, label in (
        ("command_download", "download"),
        ("command_validate", "validate"),
        ("command_prepare", "prepare-emission"),
        ("command_run", "run"),
        ("command_report", "report"),
    ):
        monkeypatch.setattr(cli, name, lambda label=label: calls.append(label))

    assert cli.main(["all"]) == 0
    assert calls == ["download", "validate", "prepare-emission", "run", "report"]


def test_module_help_lists_all_six_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ipop", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    for command in ("download", "validate", "prepare-emission", "run", "report", "all"):
        assert command in result.stdout


def test_command_prepare_rejects_unexpected_reference_ambiguity_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = pd.DataFrame({"row_id": [0]})
    audit = pd.DataFrame({"status": ["resolved"] * 5 + ["ambiguous_reference"] * 5})
    monkeypatch.setattr(cli, "load_emission_features", lambda _: prepared)
    monkeypatch.setattr(cli, "load_master_csv", lambda _: prepared)
    monkeypatch.setattr(
        cli, "attach_emission_groups", lambda _, __: MetadataJoinResult(prepared, audit)
    )
    monkeypatch.setattr(cli, "INTERIM", tmp_path)

    with pytest.raises(ValueError, match="Expected exactly 6 ambiguous source DOI rows, got 5"):
        cli.command_prepare()
