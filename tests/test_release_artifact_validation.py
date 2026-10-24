"""Exercise metadata gates used by both package publishing jobs."""

from io import BytesIO
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts.validate_release_artifacts import validate_sdist, validate_wheel


def _metadata(version="1.0.0b9", name="mm-toolbox"):
    return f"Name: {name}\nVersion: {version}\n".encode()


def _sdist(path: Path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))


def test_sdist_uses_root_metadata_and_allows_setuptools_copy(tmp_path):
    path = tmp_path / "mm_toolbox-1.0.0b9.tar.gz"
    _sdist(
        path,
        [
            ("mm_toolbox-1.0.0b9/PKG-INFO", _metadata()),
            ("mm_toolbox-1.0.0b9/src/mm_toolbox.egg-info/PKG-INFO", _metadata()),
        ],
    )
    validate_sdist(path, "1.0.0b9")


@pytest.mark.parametrize("count", [0, 2])
def test_sdist_requires_exactly_one_root_metadata(tmp_path, count):
    path = tmp_path / "package.tar.gz"
    _sdist(path, [(f"root{index}/PKG-INFO", _metadata()) for index in range(count)])
    with pytest.raises(SystemExit, match="one root PKG-INFO"):
        validate_sdist(path, "1.0.0b9")


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
@pytest.mark.parametrize("metadata", [_metadata("1.0.0b8"), _metadata(name="other")])
def test_artifacts_reject_wrong_project_or_version(tmp_path, kind, metadata):
    if kind == "sdist":
        path = tmp_path / "package.tar.gz"
        _sdist(path, [("root/PKG-INFO", metadata)])
        validate = validate_sdist
    else:
        path = tmp_path / "package.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mm_toolbox-1.0.0b9.dist-info/METADATA", metadata)
        validate = validate_wheel
    with pytest.raises(SystemExit, match="expected mm-toolbox 1.0.0b9"):
        validate(path, "1.0.0b9")


def test_wheel_accepts_normalized_project_name(tmp_path):
    path = tmp_path / "package.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mm_toolbox-1.0.0b9.dist-info/METADATA", _metadata(name="mm_toolbox")
        )
    validate_wheel(path, "1.0.0b9")


@pytest.mark.parametrize("count", [0, 2])
def test_wheel_requires_exactly_one_metadata(tmp_path, count):
    path = tmp_path / "package.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(count):
            archive.writestr(f"project{index}.dist-info/METADATA", _metadata())
    with pytest.raises(SystemExit, match="one dist-info/METADATA"):
        validate_wheel(path, "1.0.0b9")
