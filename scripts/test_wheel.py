"""Build native tests and validate an installed wheel outside the source tree."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    """Copy test sources, compile against installed headers, and run pytest."""
    project = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="mm-toolbox-wheel-tests-") as directory:
        # Preserve pytest settings and markers while keeping imports outside
        # the source checkout. Native compilation still uses installed headers.
        shutil.copy2(project / "pyproject.toml", Path(directory) / "pyproject.toml")
        tests = Path(directory) / "tests"
        shutil.copytree(
            project / "tests",
            tests,
            ignore=shutil.ignore_patterns("*.so", "*.pyd", "build", "__pycache__"),
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["MM_TOOLBOX_TEST_INSTALLED"] = "1"
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace", "--parallel", "2"],
            cwd=tests,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--require-native",
                "--maxfail=1",
                "--timeout=300",
                "-q",
                str(tests),
            ],
            cwd=directory,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
