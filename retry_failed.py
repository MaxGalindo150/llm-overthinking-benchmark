"""
retry_failed.py — Re-run only errored rows from a previous results JSON.
Patches the original file in place (keeps successful rows intact).

Usage:
    python retry_failed.py                         # uses latest results file
    python retry_failed.py --results results/results_20260222_234233.json
"""

import argparse
import json
import time
from pathlib import Path

# Re-use the same helpers from run_experiments
from run_experiments import MODELS, call_model, load_questions, detect_outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/", help="JSON file or directory")
    parser.add_argument("--benchmark", default="data/benchmark.json")
    parser.add_argument("--delay", type=float, default=1.5)
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

    failed_indices = [i for i, r in enumerate(data) if r.get("error")]
    print(f"Found {len(failed_indices)} failed rows out of {len(data)} total.\n")

    if not failed_indices:
        print("Nothing to retry.")
        return

    questions = load_questions(args.benchmark)
    q_index = {(q["id"], q["language"]): q for q in questions}

    for idx in failed_indices:
        row = data[idx]
        model_key = row["model"]
        qid = row["question_id"]
        lang = row["language"]

        q = q_index.get((qid, lang))
        if q is None:
            print(f"[SKIP] {model_key} | {qid} ({lang}) — question not found in benchmark")
            continue

        print(f"[RETRY] {model_key} | {qid} ({lang}) ...", end=" ", flush=True)
        result = call_model(model_key, q["text"], lang)

        if result.get("error"):
            print(f"STILL FAILING: {result['error'][:100]}")
        else:
            print(f"OK — detected={result['detected_impossible']}, tokens={result['output_tokens']}")

        result.update({
            "question_id": qid,
            "category": row["category"],
            "missing_premise": row["missing_premise"],
            "question_text": q["text"],
        })
        data[idx] = result

        # Save after every call
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        time.sleep(args.delay)

    # Also refresh the CSV
    import csv
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

    remaining = sum(1 for r in data if r.get("error"))
    print(f"\nDone. Remaining errors: {remaining}")
    print(f"Patched: {p}")
    print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
