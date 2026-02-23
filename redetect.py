"""
redetect.py — Re-run detection logic on existing results without API calls.
Patches detected_impossible and abstained fields in place.

Usage:
    python redetect.py
    python redetect.py --results results/results_20260222_234233.json
"""

import argparse
import csv
import json
from pathlib import Path

from run_experiments import detect_outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/", help="JSON file or directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p = Path(args.results)
    if p.is_dir():
        files = sorted(p.glob("results_*.json"))
        if not files:
            raise FileNotFoundError(f"No results_*.json in {p}")
        p = files[-1]

    print(f"Loading: {p}")
    with open(p, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    flipped = 0
    for r in data:
        if r.get("error") or not r.get("text"):
            continue
        old_detected = r["detected_impossible"]
        old_abstained = r["abstained"]
        new_detected, new_abstained = detect_outcome(r["text"], r["language"])
        if new_detected != old_detected or new_abstained != old_abstained:
            flipped += 1
            print(
                f"  {r['model']:25} {r['question_id']} ({r['language']}): "
                f"detected {old_detected} → {new_detected}, "
                f"abstained {old_abstained} → {new_abstained}"
            )
        r["detected_impossible"] = new_detected
        r["abstained"] = new_abstained

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    csv_path = p.with_suffix(".csv")
    fieldnames = [
        "question_id", "category", "language", "model",
        "detected_impossible", "abstained",
        "output_tokens", "input_tokens", "cost_usd", "latency_s", "error",
        "text",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)

    print(f"\n{flipped} rows updated. Saved: {p}")


if __name__ == "__main__":
    main()
