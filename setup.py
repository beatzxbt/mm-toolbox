from pathlib import Path
from typing import List

from setuptools import setup, find_packages

from mm_toolbox import __VERSION__, clean_repo


def readme():
    with open(str(Path(__file__).parent.resolve()) + "/README.md") as f:
        return f.read()


def read_requirements() -> List[str]:
    return open(str(Path(__file__).parent.resolve()) + "/requirements.txt").readlines()


NAME = "mm_toolbox"
VERSION = __VERSION__
DESCRIPTION = "python toolbox of fast mm-related funcs"
LONG_DESCRIPTION = readme()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/beatzxbt/mm-toolbox"
AUTHOR = "beatzxbt"
LICENSE = "MIT"
PACKAGES = find_packages()
INCLUDE_PACKAGE_DATA = True
ZIP_SAFE = False
REQUIREMENTS = read_requirements()

if __name__ == "__main__":
    setup(
        name=NAME,
        version=__VERSION__,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        url=URL,
        author=AUTHOR,
        license=LICENSE,
        packages=PACKAGES,
        include_package_data=INCLUDE_PACKAGE_DATA,
        zip_safe=ZIP_SAFE,
        install_requires=REQUIREMENTS,
    )
    clean_repo(NAME)
