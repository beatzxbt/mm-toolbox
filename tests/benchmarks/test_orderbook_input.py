"""Orderbook benchmark CLIs must fail clearly when capture data is missing."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script", "extra_args"),
    [
        ("benchmark_standard_orderbook.py", []),
        ("benchmark_advanced_orderbook.py", []),
        ("benchmark_advanced_orderbook.py", ["--multi-size"]),
    ],
)
def test_missing_orderbook_capture_exits_nonzero(tmp_path, script, extra_args):
    project = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "missing.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(project / "benchmarks" / "orderbook" / script),
            "--input",
            str(input_path),
            "--operations",
            "2",
            "--warmup",
            "0",
            *extra_args,
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    assert f"Input file not found: {input_path}" in result.stderr
    assert "collect_data.py --output <path>" in result.stderr
    assert "--input" in result.stderr
