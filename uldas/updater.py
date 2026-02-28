#file: uldas/updater.py

import logging
import requests
from packaging import version as pkg_version

from uldas.constants import VERSION

logger = logging.getLogger(__name__)


def check_for_updates() -> None:
    try:
        print("Checking for updates...", end=" ", flush=True)
        url = "https://api.github.com/repos/netplexflix/ULDAS/releases/latest"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        latest = resp.json().get("tag_name", "").lstrip("v")
        if not latest:
            print("Could not determine latest version")
            return
        try:
            if pkg_version.parse(latest) > pkg_version.parse(VERSION):
                print("UPDATE AVAILABLE!")
                print(f"\n{'=' * 60}")
                print("📄 UPDATE AVAILABLE")
                print(f"{'=' * 60}")
                print(f"Current version: {VERSION}")
                print(f"Latest version:  {latest}")
                print("Download from: https://github.com/netplexflix/ULDAS")
                print(f"{'=' * 60}\n")
            else:
                print(f"✓ Up to date. Version: {VERSION}")
        except Exception:
            if latest != VERSION:
                print(f"Update may be available (current: {VERSION}, latest: {latest})")
            else:
                print(f"✓ Up to date. Version: {VERSION}")
    except requests.exceptions.RequestException:
        print("Failed (network error)")
    except Exception:
        print("Failed (error)")