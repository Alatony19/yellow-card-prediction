"""Run the Premier League pipeline end to end on synthetic data, then its leakage gates.

    python tools/demo.py

Generates ``data_synthetic/`` if it is not there, parses it with the real parser, builds
the leakage-free feature table, and runs ``tests/test_pipeline.py``. Real data and real
artifacts are never touched: ``YCP_DATA_DIR`` / ``YCP_ARTIFACT_DIR`` point the whole
pipeline at ``data_synthetic/`` and ``artifacts_synthetic/`` for these steps only.

Takes about ten minutes. The numbers it prints describe an invented league, not real football.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_synthetic"
ARTIFACTS = ROOT / "artifacts_synthetic"

env = dict(os.environ, YCP_DATA_DIR=str(DATA), YCP_ARTIFACT_DIR=str(ARTIFACTS),
           PYTHONIOENCODING="utf-8")


def step(title: str, *cmd: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))
    t0 = time.time()
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env)
    print(f"--- {title}: {'ok' if r.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f}s)")
    if r.returncode:
        sys.exit(r.returncode)


if not (DATA / "fbref").exists():
    step("1/6 generate synthetic archive", "tools/make_synthetic_data.py", "--out", str(DATA))
else:
    print("1/6 synthetic archive already present - reusing", DATA)
step("2/6 parse match folders -> base tables", "pipeline/build_base_table.py")
step("3/6 build the leakage-free feature table", "pipeline/features.py")
step("4/6 integrity and leakage gates", "tests/test_pipeline.py")
step("5/6 match-result table (goals, ratings, closing lines)", "match_model/build_match_table.py")
step("6/6 match-model gates", "match_model/tests/test_match_model.py")
print("\nAll gates passed on the synthetic archive.")
