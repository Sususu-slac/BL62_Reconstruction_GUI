from __future__ import annotations

import re
from pathlib import Path


_DRIVE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def windows_to_wsl(path: str | Path) -> str:
    value = str(path)
    match = _DRIVE.match(value)
    if match:
        rest = match.group("rest").replace("\\", "/")
        return f"/mnt/{match.group('drive').lower()}/{rest}"
    if value.startswith("\\\\wsl.localhost\\"):
        parts = value.split("\\")
        return "/" + "/".join(parts[4:])
    return value.replace("\\", "/")


def wsl_to_windows(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", value)
    if match:
        return f"{match.group(1).upper()}:\\{match.group(2).replace('/', chr(92))}"
    return value
