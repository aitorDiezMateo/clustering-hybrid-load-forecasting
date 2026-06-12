"""
Hybrid Short-Term Load Forecasting (STLF) pipeline.

Usage:
    python main.py
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _load_src_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_stage(name: str, script_path: Path, config: dict) -> None:
    log.info("=" * 60)
    log.info("STAGE START: %s", name)
    log.info("=" * 60)
    t0 = time.perf_counter()
    try:
        mod = _load_src_module(script_path)
        mod.run(config)
        elapsed = time.perf_counter() - t0
        log.info("STAGE DONE:  %s  [%.1fs]", name, elapsed)
    except Exception as exc:
        log.error("STAGE FAILED: %s  [%.1fs]  — %s", name, time.perf_counter() - t0, exc, exc_info=True)
        raise SystemExit(1) from exc


def main() -> None:
    project_root = Path(__file__).parent.resolve()
    config = {"_project_root": str(project_root)}

    src = project_root / "src"
    stages = [
        ("01 — Interval Clustering", src / "01_intervals_clustering.py"),
        ("02 — Day Classifier",      src / "02_day_classifier.py"),
        ("03 — Max Load Regressor",  src / "03_max_load_regressor.py"),
        ("04 — Predictions",         src / "04_predictions.py"),
        ("05 — Benchmark",           src / "05_benchmark.py"),
    ]

    t_total = time.perf_counter()
    log.info("Pipeline has %d stages", len(stages))

    for stage_name, script_path in stages:
        _run_stage(stage_name, script_path, config)

    total_elapsed = time.perf_counter() - t_total
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE  [%.1fs total]", total_elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
