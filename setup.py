import pathlib
import re

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent
readme_file = here / "README.rst"
source_file = here / "src" / "rr" / "channels.py"
version_match = re.search(r"__version__\s*=\s*(['\"])(.*)\1", source_file.read_text())
if version_match is None:
    raise Exception("unable to extract version from {}".format(source_file))
version = version_match.group(2)
packages = ["rr." + p for p in find_packages("src/rr")] or ["rr"]

setup(
    name="rr.channels",
    version=version,
    description="Yet another take at the observer pattern in Python.",
    long_description=readme_file.read_text(),
    url="https://github.com/2xR/rr.channels",
    author="Rui Jorge Rei",
    author_email="rui.jorge.rei@googlemail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    packages=packages,
    package_dir={"": "src"},
)
