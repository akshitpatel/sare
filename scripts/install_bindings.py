#!/usr/bin/env python3
"""
Install the built C++ extension module into the in-repo Python package.

Why this exists:
- Overwriting an already-imported, code-signed Mach-O binary in-place can lead
  to macOS code-signing cache issues and the importing Python process being
  SIGKILL'd.
- We avoid that by copying to a temp file and atomically replacing the target.

Usage:
  python3 scripts/install_bindings.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
PKG_DIR = REPO_ROOT / "python" / "sare"


def _find_built_module() -> Path:
    # Typical filename: sare_bindings.cpython-313-darwin.so
    candidates = sorted(BUILD_DIR.glob("sare_bindings*.so"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No built module found at {BUILD_DIR}/sare_bindings*.so")
    return candidates[0]


def _atomic_install(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)  # atomic on same filesystem


def main() -> int:
    src = _find_built_module()
    dst = PKG_DIR / src.name
    _atomic_install(src, dst)
    print(f"Installed {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

