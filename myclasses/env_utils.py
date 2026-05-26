"""
env_utils.py — Cross-platform environment detection utilities.

Provides a single, importable way to detect the runtime environment
(Google Colab, local macOS, local Linux) so the rest of the codebase
can branch cleanly without scattered platform checks.
"""

import os
import platform
import subprocess
import sys


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    return "google.colab" in sys.modules or "COLAB_BACKEND_VERSION" in os.environ


def is_macos() -> bool:
    """Return True if running on macOS."""
    return platform.system() == "Darwin"


def is_linux() -> bool:
    """Return True if running on Linux (includes Colab)."""
    return platform.system() == "Linux"


def is_jupyter() -> bool:
    """Return True if running inside any Jupyter kernel (Colab, JupyterLab, VS Code, etc.)."""
    return "ipykernel" in sys.modules


def get_cpu_info() -> str:
    """
    Return a human-readable CPU info string.
    Uses `lscpu` on Linux/Colab and `sysctl` on macOS.
    Falls back gracefully if neither command is available.
    """
    if is_linux():
        result = subprocess.run(
            ["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8")
        return "(lscpu not available)"

    if is_macos():
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return f"CPU: {result.stdout.decode().strip()}\n"
        return "(sysctl not available)"

    return f"(CPU info not supported on {platform.system()})"


def mount_colab_drive(mount_point: str = "/content/drive") -> bool:
    """
    Mount Google Drive when running in Colab.
    No-op (returns False) when running locally — so callers can safely
    call this unconditionally without any try/except at call site.

    Returns:
        True  — drive was mounted (Colab)
        False — skipped (local run)
    """
    if not is_colab():
        print("[env_utils] Not in Colab — skipping Drive mount.")
        return False

    from google.colab import drive  # type: ignore[import]
    drive.mount(mount_point)
    return True
