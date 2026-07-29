r"""
Apply the Smart App Control workaround: overwrite the venv's blocked
uuid_utils/__init__.py with our pure-Python version.

Run with the venv's Python:
    .\.venv\Scripts\python.exe sac_workaround\apply_sac_workaround.py

Idempotent. Safe to re-run after rebuilding the venv.
"""

import shutil
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLACEMENT = HERE / "uuid_utils_init.py"


def main() -> int:
    # Locate the installed package WITHOUT importing it (the import is what's
    # blocked). site-packages is where pip installs into for this interpreter.
    site_packages = Path(sysconfig.get_paths()["purelib"])
    target = site_packages / "uuid_utils" / "__init__.py"
    if not target.exists():
        print(f"ERROR: {target} not found. Is uuid_utils installed in this venv?")
        return 1

    backup = target.with_suffix(".py.orig")

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backed up original -> {backup}")

    shutil.copy2(REPLACEMENT, target)
    print(f"Installed pure-Python shim -> {target}")
    print("Done. Re-run this after any `pip install`/venv rebuild that touches uuid_utils.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
