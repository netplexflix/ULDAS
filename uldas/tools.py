# file: uldas/tools.py

import os
import sys
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def find_executable(name: str) -> Optional[str]:
    """Return the path to *name* (or *name*.exe on Windows), or ``None``."""
    if shutil.which(name):
        return name

    if sys.platform == "win32":
        exe_name = f"{name}.exe"
        if shutil.which(exe_name):
            return exe_name

        if name == "mkvpropedit":
            candidates = [
                r"C:\Program Files\MKVToolNix\mkvpropedit.exe",
                r"C:\Program Files (x86)\MKVToolNix\mkvpropedit.exe",
                r"C:\ProgramData\chocolatey\lib\mkvtoolnix\tools\mkvpropedit.exe",
                r"C:\MKVToolNix\mkvpropedit.exe",
                r"C:\Tools\MKVToolNix\mkvpropedit.exe",
            ]
            for path in candidates:
                if os.path.exists(path):
                    logger.info("Found mkvpropedit at: %s", path)
                    return path

            for pf in (r"C:\Program Files", r"C:\Program Files (x86)"):
                if os.path.isdir(pf):
                    for item in os.listdir(pf):
                        if "mkv" in item.lower():
                            p = os.path.join(pf, item, "mkvpropedit.exe")
                            if os.path.exists(p):
                                logger.info("Found mkvpropedit at: %s", p)
                                return p

        common = [
            rf"C:\Program Files\FFmpeg\bin\{exe_name}",
            rf"C:\Program Files (x86)\FFmpeg\bin\{exe_name}",
        ]
        for path in common:
            if os.path.exists(path):
                return path

    return None


def find_mkvtoolnix_installation() -> Optional[str]:
    """Windows-only helper to locate MKVToolNix via the registry or filesystem."""
    if sys.platform != "win32":
        return None

    print("Searching for MKVToolNix installation...")

    # ── Registry search ──────────────────────────────────────────────────
    try:
        import winreg

        reg_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        for rp in reg_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rp) as key:
                    i = 0
                    while True:
                        try:
                            sub_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, sub_name) as sub:
                                try:
                                    display = winreg.QueryValueEx(sub, "DisplayName")[0]
                                    if "mkvtoolnix" in display.lower():
                                        try:
                                            loc = winreg.QueryValueEx(sub, "InstallLocation")[0]
                                            print(f"Found MKVToolNix installed at: {loc}")
                                            exe = os.path.join(loc, "mkvpropedit.exe")
                                            if os.path.exists(exe):
                                                print(f"mkvpropedit.exe found at: {exe}")
                                                return exe
                                        except FileNotFoundError:
                                            pass
                                except FileNotFoundError:
                                    pass
                            i += 1
                        except OSError:
                            break
            except OSError:
                continue
    except ImportError:
        pass

    # ── Filesystem search ────────────────────────────────────────────────
    for base in (r"C:\Program Files", r"C:\Program Files (x86)",
                 r"C:\ProgramData\chocolatey\lib"):
        if not os.path.isdir(base):
            continue
        for item in os.listdir(base):
            if "mkv" not in item.lower():
                continue
            full = os.path.join(base, item)
            print(f"Found MKV-related directory: {full}")
            for sub in ("", "tools", "bin"):
                exe = os.path.join(full, sub, "mkvpropedit.exe") if sub else os.path.join(full, "mkvpropedit.exe")
                if os.path.exists(exe):
                    print(f"mkvpropedit.exe found at: {exe}")
                    return exe

    return None