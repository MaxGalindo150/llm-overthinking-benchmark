"""
Missing Premise Benchmark — Experiment Runner
=============================================
Sends all 40 benchmark questions to multiple LLM APIs and records:
  - tokens generated
  - whether the model detected the impossible/missing premise
  - whether the model abstained (refused to answer)
  - estimated cost

Usage:
    python run_experiments.py [--models MODEL1 MODEL2 ...] [--dry-run]

Requirements:
    pip install openai anthropic google-generativeai requests python-dotenv tqdm
"""

import json
import os
import time
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ─── Model registry ───────────────────────────────────────────────────────────

MODELS: dict[str, dict[str, Any]] = {
    "deepseek-r1": {
        "provider": "deepseek",
        "model_id": "deepseek-reasoner",
        "cost_per_1k_input": 0.00055,   # USD (cache miss)
        "cost_per_1k_output": 0.00219,
        "is_reasoning": True,
    },
    "qwen3-thinking": {
        "provider": "openrouter",
        "model_id": "qwen/qwq-32b",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.00060,
        "is_reasoning": True,
    },
    "gpt-o3-mini": {
        "provider": "openai",
        "model_id": "o3-mini",
        "cost_per_1k_input": 0.00110,
        "cost_per_1k_output": 0.00440,
        "is_reasoning": True,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "cost_per_1k_input": 0.00300,
        "cost_per_1k_output": 0.01500,
        "is_reasoning": False,
    },
    "gemini-flash-thinking": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
        "cost_per_1k_input": 0.000150,   # Gemini 2.5 Flash list price
        "cost_per_1k_output": 0.000600,
        "is_reasoning": True,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "cost_per_1k_input": 0.00250,
        "cost_per_1k_output": 0.01000,
        "is_reasoning": False,
    },
}

# ─── Detection heuristics ─────────────────────────────────────────────────────

DETECTION_KEYWORDS = {
    "es": [
        "no se puede", "imposible", "falta", "insuficiente", "no hay suficiente",
        "no tengo suficiente", "información incompleta", "datos faltantes",
        "no es posible", "no puedo determinar", "no se puede calcular",
        "premisa faltante", "dato desconocido", "no se especifica",
        "no está definido", "indeterminado", "contradicción",
        "no se puede resolver", "no hay solución", "irresoluble",
        "no se puede responder", "no puedo responder",
    ],
    "en": [
        "cannot", "can't", "impossible", "missing", "insufficient",
        "not enough information", "not enough data", "incomplete",
        "cannot be determined", "cannot calculate", "missing premise",
        "unknown", "not specified", "not defined", "indeterminate",
        "contradiction", "cannot be solved", "no solution", "unsolvable",
        "cannot answer", "unable to answer", "unable to determine",
        "no way to", "there is no",
    ],
}

ABSTENTION_KEYWORDS = {
    "es": [
        "no puedo responder", "me abstendré", "no voy a responder",
        "no responderé", "prefiero no", "declino",
    ],
    "en": [
        "i cannot answer", "i'll abstain", "i won't answer",
        "i refuse", "i decline", "i prefer not",
    ],
}


def detect_outcome(response_text: str, language: str = "") -> tuple[bool, bool]:  # noqa: ARG001
    """
    Returns (detected_impossible, abstained).
    detected_impossible: model correctly identifies missing/contradictory premise.
    abstained: model refuses without explanation.
    Always searches both ES and EN keywords — models often reply in English
    regardless of the question language.
    """
    text = response_text.lower()
    all_detection = DETECTION_KEYWORDS["es"] + DETECTION_KEYWORDS["en"]
    all_abstention = ABSTENTION_KEYWORDS["es"] + ABSTENTION_KEYWORDS["en"]

    detected = any(kw in text for kw in all_detection)
    abstained = (not detected) and any(kw in text for kw in all_abstention)
    return detected, abstained


# ─── API clients ──────────────────────────────────────────────────────────────

def call_openai(model_id: str, prompt: str, system: str) -> dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    # o3-mini uses max_completion_tokens, not max_tokens
    if model_id.startswith("o"):
        kwargs["max_completion_tokens"] = 4096
    else:
        kwargs["max_tokens"] = 1024

    resp = client.chat.completions.create(**kwargs)
    return {
        "text": resp.choices[0].message.content or "",
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def call_anthropic(model_id: str, prompt: str, system: str) -> dict[str, Any]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    resp = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": resp.content[0].text,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


def call_google(model_id: str, prompt: str, system: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=8192,
        ),
    )
    usage = resp.usage_metadata
    return {
        "text": resp.text,
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
    }


def call_deepseek(model_id: str, prompt: str, system: str) -> dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
    )
    return {
        "text": resp.choices[0].message.content or "",
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def call_openrouter(model_id: str, prompt: str, system: str) -> dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
    )
    return {
        "text": resp.choices[0].message.content or "",
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


PROVIDER_DISPATCH = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "deepseek": call_deepseek,
    "openrouter": call_openrouter,
}

SYSTEM_PROMPT = (
    "You are a critical reasoning assistant. "
    "When presented with a question, carefully evaluate whether it contains all the "
    "necessary information to produce a definitive answer. "
    "If key information is missing or the question contains a logical contradiction, "
    "clearly state that the question cannot be answered and explain why. "
    "Do not make assumptions or invent missing data."
)


def call_model(model_key: str, prompt: str, language: str) -> dict[str, Any]:
    cfg = MODELS[model_key]
    fn = PROVIDER_DISPATCH[cfg["provider"]]

    start = time.time()
    try:
        result = fn(cfg["model_id"], prompt, SYSTEM_PROMPT)
        elapsed = time.time() - start
    except Exception as exc:
        return {
            "model": model_key,
            "language": language,
            "error": str(exc),
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_s": round(time.time() - start, 2),
            "cost_usd": 0.0,
            "detected_impossible": False,
            "abstained": False,
        }

    cost = (
        result["input_tokens"] / 1000 * cfg["cost_per_1k_input"]
        + result["output_tokens"] / 1000 * cfg["cost_per_1k_output"]
    )
    detected, abstained = detect_outcome(result["text"], language)

    return {
        "model": model_key,
        "language": language,
        "error": None,
        "text": result["text"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_s": round(elapsed, 2),
        "cost_usd": round(cost, 6),
        "detected_impossible": detected,
        "abstained": abstained,
    }


# ─── Question loading ──────────────────────────────────────────────────────────

def load_questions(benchmark_path: str) -> list[dict[str, Any]]:
    """Flatten benchmark JSON into a list of (question_id, language, text) dicts."""
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)

    flat: list[dict[str, Any]] = []
    for q in data["questions"]:
        for lang in ("es", "en"):
            # Handle override fields (e.g. override_es)
            override_key = f"override_{lang}"
            if override_key in q:
                text = q[override_key]["text"]
                answer_impossible = q[override_key]["answer_impossible"]
            else:
                text = q["variants"][lang]["text"]
                answer_impossible = q["variants"][lang]["answer_impossible"]

            flat.append({
                "id": q["id"],
                "category": q["category"],
                "language": lang,
                "text": text,
                "answer_impossible": answer_impossible,
                "missing_premise": q["missing_premise"],
            })
    return flat


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Missing Premise Benchmark Runner")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which models to evaluate (default: all)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["es", "en"],
        default=["es", "en"],
        help="Which language variants to test (default: both)",
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark.json",
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (rate limit buffer)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be sent without making API calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions = [
        q for q in load_questions(args.benchmark)
        if q["language"] in args.languages
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    csv_file = output_dir / f"results_{timestamp}.csv"

    all_results: list[dict[str, Any]] = []

    total = len(args.models) * len(questions)
    print(f"\nRunning {len(questions)} questions × {len(args.models)} models = {total} API calls\n")

    with tqdm(total=total, unit="call") as pbar:
        for model_key in args.models:
            for q in questions:
                pbar.set_description(f"{model_key[:20]} | {q['id']} ({q['language']})")

                if args.dry_run:
                    print(f"[DRY RUN] {model_key} | {q['id']} ({q['language']}): {q['text'][:80]}...")
                    pbar.update(1)
                    continue

                row = call_model(model_key, q["text"], q["language"])
                row.update({
                    "question_id": q["id"],
                    "category": q["category"],
                    "missing_premise": q["missing_premise"],
                    "question_text": q["text"],
                })
                all_results.append(row)

                # Stream-save after each call in case of interruption
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

                pbar.update(1)
                time.sleep(args.delay)

    if args.dry_run:
        print("\nDry run complete. No API calls were made.")
        return

    # Save CSV summary
    if all_results:
        fieldnames = [
            "question_id", "category", "language", "model",
            "detected_impossible", "abstained",
            "output_tokens", "input_tokens", "cost_usd", "latency_s", "error",
            "text",
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\nResults saved to:\n  JSON: {results_file}\n  CSV:  {csv_file}")

    # Quick summary
    print("\n── Quick Summary ──────────────────────────────────────────")
    from collections import defaultdict
    by_model: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    print(f"{'Model':<25} {'Detection%':>10} {'Abstain%':>10} {'Avg tokens':>12} {'Total $':>10}")
    print("─" * 70)
    for model, rows in sorted(by_model.items()):
        valid = [r for r in rows if not r.get("error")]
        if not valid:
            print(f"{model:<25} {'ERROR':>10}")
            continue
        det_pct = 100 * sum(r["detected_impossible"] for r in valid) / len(valid)
        abs_pct = 100 * sum(r["abstained"] for r in valid) / len(valid)
        avg_tok = sum(r["output_tokens"] for r in valid) / len(valid)
        total_cost = sum(r["cost_usd"] for r in valid)
        print(f"{model:<25} {det_pct:>9.1f}% {abs_pct:>9.1f}% {avg_tok:>12.0f} ${total_cost:>9.4f}")


if __name__ == "__main__":
    main()
