import json
from pathlib import Path

from ipop.cli import build_parser, write_validation_report
from ipop.data import DatasetSummary


def test_parser_exposes_six_commands() -> None:
    parser = build_parser()
    for command in ("download", "validate", "prepare-emission", "run", "report", "all"):
        args = parser.parse_args([command])
        assert args.command == command


def test_validation_report_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    write_validation_report(DatasetSummary(3952, 2238, 553, 16023), path)
    assert json.loads(path.read_text(encoding="utf-8"))["target_observations"] == 16023
