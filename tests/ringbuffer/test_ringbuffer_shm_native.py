"""Exercise native SHM race windows without depending on Python scheduling."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shared memory")


@pytest.fixture(scope="module")
def native_shm_executable(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("Native SHM harness requires a C compiler")
    root = Path(__file__).resolve().parents[2]
    if os.environ.get("MM_TOOLBOX_TEST_INSTALLED") == "1":
        import mm_toolbox

        source = Path(mm_toolbox.__file__).resolve().parent
    else:
        source = root / "src/mm_toolbox"
    executable = tmp_path_factory.mktemp("native-shm") / "regression"
    command = [
        compiler,
        "-std=c11",
        "-D_POSIX_C_SOURCE=200809L",
        "-DSHM_TEST_AFTER_READ_WORD=shm_test_after_read_word",
        "-O2",
        "-pthread",
        "-I",
        str(source / "ringbuffer/shm/c"),
        "-I",
        str(source / "ringbuffer/shm"),
        str(Path(__file__).with_name("shm_native_regression.c")),
        str(source / "ringbuffer/shm/c/shm_core.c"),
        str(source / "ringbuffer/shm/c/shm_helpers.c"),
        str(source / "time/ctime_impl.c"),
        "-lm",
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return executable


def test_native_shm_interleavings_and_overflow(native_shm_executable):
    result = subprocess.run(
        [str(native_shm_executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "deterministic interleavings" in result.stdout
    assert "concurrency passed" in result.stdout
