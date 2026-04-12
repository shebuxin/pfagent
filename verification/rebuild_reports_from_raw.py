from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verification.reporting import generate_reports


def _load_scenarios_for_model(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw_root = run_dir / "raw"
    if not raw_root.exists():
        return {}

    results: Dict[str, List[Dict[str, Any]]] = {}
    for model_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        scenarios: List[Dict[str, Any]] = []
        for scenario_path in sorted(model_dir.glob("*/scenario_result.json")):
            with scenario_path.open(encoding="utf-8") as handle:
                scenario = json.load(handle)
            for index, turn in enumerate(scenario.get("turns", []), start=1):
                turn.setdefault("turn_id", index)
            scenarios.append(scenario)
        if scenarios:
            results[model_dir.name] = scenarios
    return results


def rebuild_results(run_dirs: List[Path], output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    combined_results: Dict[str, Any] = {
        "run_root": str(output_root),
        "scenario_count": 0,
        "models": {},
    }

    for run_dir in run_dirs:
        model_results = _load_scenarios_for_model(run_dir)
        for model_key, scenarios in model_results.items():
            combined_results["models"][model_key] = scenarios
            combined_results["scenario_count"] = max(combined_results["scenario_count"], len(scenarios))

    results_path = output_root / "verification_results_rebuilt.json"
    results_path.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
    combined_results["results_path"] = str(results_path)
    combined_results["report_paths"] = generate_reports(combined_results, output_root)
    (output_root / "verification_manifest_rebuilt.json").write_text(
        json.dumps(combined_results, indent=2),
        encoding="utf-8",
    )
    return combined_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild verification summary reports from raw scenario_result artifacts.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more run directories that contain raw/<model>/.../scenario_result.json files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("verification") / "artifacts_parallel" / f"rebuilt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Directory for the rebuilt combined report.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    combined = rebuild_results([Path(path) for path in args.run_dirs], Path(args.output_root))
    print(f"Rebuilt results: {combined['results_path']}")
    print(f"Summary report: {combined['report_paths']['summary_markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
