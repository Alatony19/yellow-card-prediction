"""Put every code folder of v3 on ``sys.path``.

The modules import each other flat (``import config``, ``import predict``) and
live in three folders by role - ``pipeline/`` (data -> features -> models),
``serve/`` (scoring fixtures) and ``research/`` (the experiments behind the
design).  Every entry point adds v3 itself to ``sys.path`` and imports this
module, so a script runs the same from any folder and any working directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _d in (ROOT, ROOT / "research", ROOT / "serve", ROOT / "pipeline"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
