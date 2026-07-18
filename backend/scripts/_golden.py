"""Shared golden-set loader with a freeze guard (R3-1).

Recomputes SHA256 of items[] and refuses to run if it differs from the frozen
items_sha256, so a silent edit to the frozen set is auto-detected at eval start.
"""
import hashlib
import json
from pathlib import Path


def items_hash(items) -> str:
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_frozen():
    p = Path(__file__).resolve().parents[2] / "results" / "golden_set_v1.json"
    g = json.load(open(p, encoding="utf-8"))
    h = items_hash(g["items"])
    frozen = g.get("items_sha256")
    if frozen and h != frozen:
        raise SystemExit(
            f"GOLDEN SET FREEZE VIOLATION: items hash {h} != frozen {frozen}.\n"
            "The frozen golden set was edited. Re-ratify and re-freeze before evaluating."
        )
    return g


def hardware_stamp() -> str:
    """Accurate OS stamp (R3-2): platform.release() reports '10' on Windows 11;
    the build number disambiguates (>= 22000 is Windows 11)."""
    import platform
    sysname = platform.system()
    if sysname == "Windows":
        ver = platform.version()  # e.g. 10.0.26200
        build = 0
        try:
            build = int(ver.split(".")[-1])
        except Exception:
            pass
        win = "Windows 11" if build >= 22000 else "Windows 10"
        return f"{win} (build {ver}) | {platform.machine()} | py{platform.python_version()}"
    return f"{sysname} {platform.release()} | {platform.machine()} | py{platform.python_version()}"
