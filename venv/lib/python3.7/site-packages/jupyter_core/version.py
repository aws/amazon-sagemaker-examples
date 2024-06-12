# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from collections import namedtuple

VersionInfo = namedtuple("VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"])

version_info = VersionInfo(4, 12, 0, "final", 0)
__version__ = "4.12.0"

_specifier_ = {"alpha": "a", "beta": "b", "candidate": "rc", "final": "", "dev": "dev"}

if version_info.releaselevel == "final":
    _suffix_ = ""
elif version_info.releaselevel == "dev":
    _suffix_ = f".dev{version_info.serial}"
else:
    _suffix_ = f"{_specifier_[version_info.releaselevel]}{version_info.serial}"

assert __version__ == f"{version_info.major}.{version_info.minor}.{version_info.micro}{_suffix_}"
