from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lidar_system_algorithm.run_tensorrt_bucketed_core_wrapper import main as run_main  # noqa: E402


if __name__ == "__main__":
    if "--build-only" not in sys.argv:
        sys.argv.append("--build-only")
    run_main()
