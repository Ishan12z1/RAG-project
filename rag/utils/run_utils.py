from __future__ import annotations
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Toronto")

def new_run_id() -> str:
    return datetime.now(TZ).strftime("%Y%m%d_%H%M%S")

def runs_root() -> Path:
    return Path("data") / "runs"

def run_dir(run_id: str) -> Path:
    return runs_root() / run_id

def latest_run_dir() -> Path:
    root = runs_root()
    runs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError("No runs under data/runs")
    return runs[-1]

def resolve_run_dir(run_id: str | None, create: bool = False) -> Path:
    base = run_dir(run_id) if run_id else latest_run_dir()
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base