import os
from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh

repo_base_dir = sh.git("rev-parse", "--show-toplevel").strip()


def run_sh_command(command: List[str]):
    """Default method for executing shell commands with pytest and sh package."""
    msg = None
    try:
        repo_base_dir = sh.git("rev-parse", "--show-toplevel").strip()
        sh.python(command, _cwd=repo_base_dir)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)
