"""Validate downloaded release artifacts against the selected release version."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def check_metadata(metadata: bytes, artifact: Path, expected_version: str) -> None:
    """Require an artifact to contain the expected project name and version."""
    parsed = BytesParser().parsebytes(metadata)
    name = parsed.get("Name", "").replace("_", "-").lower()
    version = parsed.get("Version", "")
    if name != "mm-toolbox" or version != expected_version:
        raise SystemExit(
            f"{artifact.name} has distribution metadata {name} {version}; "
            f"expected mm-toolbox {expected_version}."
        )


def validate_wheel(wheel: Path, expected_version: str) -> None:
    """Validate the sole wheel metadata file."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise SystemExit(f"{wheel.name} must contain one dist-info/METADATA file.")
        check_metadata(archive.read(metadata_paths[0]), wheel, expected_version)


def validate_sdist(sdist: Path, expected_version: str) -> None:
    """Validate the canonical root PKG-INFO, ignoring egg-info copies."""
    with tarfile.open(sdist, "r:gz") as archive:
        metadata_members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        if len(metadata_members) != 1:
            raise SystemExit(
                f"{sdist.name} must contain one root PKG-INFO file; "
                f"found {len(metadata_members)}."
            )
        metadata_file = archive.extractfile(metadata_members[0])
        if metadata_file is None:
            raise SystemExit(f"Could not read PKG-INFO from {sdist.name}.")
        check_metadata(metadata_file.read(), sdist, expected_version)


def main() -> None:
    """Check that each downloaded wheel and source distribution matches a tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="Expected project version without the v prefix",
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    expected_version = args.version
    if not expected_version or expected_version.startswith("v"):
        parser.error(
            "--version must be a non-empty project version without a leading v"
        )

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if not wheels or not sdists:
        raise SystemExit("Expected at least one wheel and one source distribution.")

    for wheel in wheels:
        validate_wheel(wheel, expected_version)
    for sdist in sdists:
        validate_sdist(sdist, expected_version)

    print(
        f"Validated {len(wheels)} wheel(s) and {len(sdists)} sdist(s) "
        f"for {expected_version}."
    )


if __name__ == "__main__":
    main()
