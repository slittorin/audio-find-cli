#!/usr/bin/env python
"""
Bootstrap script for Windows.

Hardcoded to Python 3.11, since we otherwise get errors with some of the packages.

Run (recommended):
  py -3.11 build.py
(or ensure your `python` is 3.11)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        check=True,
        stdin=subprocess.DEVNULL,  # prevents waiting for input
    )


def ensure_py311() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"This script must be run with Python 3.11. "
            f"Current: {sys.version_info.major}.{sys.version_info.minor}\n\n"
            f"Tip: run:  py -3.11 build.py"
        )


def ensure_venv() -> Path:
    """Return a usable python inside .venv, creating/recreating if needed."""
    candidates = [
        VENV_DIR / "Scripts" / "python.exe",  # Windows default
        VENV_DIR / "Scripts" / "python",      # In case the exe suffix is missing
        VENV_DIR / "bin" / "python",          # Posix-style layout
    ]

    for path in candidates:
        if path.exists():
            return path

    # If .venv exists but has no python for this platform, rebuild it with the current interpreter.
    if VENV_DIR.exists():
        print("Recreating .venv for this platform...")

    run([sys.executable, "-m", "venv", str(VENV_DIR)])

    for path in candidates:
        if path.exists():
            return path

    raise SystemExit("Failed to locate python in .venv after creation. Please remove .ve 
def pip_install(python: Path, *args: str) -> None:
    # Always use venv python, so we don't depend on "activate" in the parent shell.
    run([str(python), "-m", "pip", "install", *args])


def main() -> None:
    ensure_py311()
    venv_python = ensure_venv()

    # Upgrade core packaging tools
    #pip_install(venv_python, "-U", "pip", "setuptools", "wheel")

    # Install required package(s)
    #pip_install(venv_python, "torchcodec")

    # Install current project in editable mode
    pip_install(venv_python, "-e", ".")

    print("\n✅ Done. Virtual environment is ready and project installed (editable).")
    print(f"Activate in PowerShell:\n  .\.venv\Scripts\\activate")

if __name__ == "__main__":
    main()
