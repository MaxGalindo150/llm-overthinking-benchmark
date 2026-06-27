"""
Missing Premise Benchmark — Analysis & Visualization
=====================================================
Reads results JSON and generates:
  1. Bar chart: detection rate by model
  2. Bar chart: avg output tokens by model
  3. Scatter plot: tokens vs detection rate (efficiency frontier)
  4. Heatmap: detection rate by model × category
  5. Summary CSV table

Usage:
    python analyze.py --results results/results_YYYYMMDD_HHMMSS.json
    python analyze.py --results results/  # uses latest file in directory
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── Styling ──────────────────────────────────────────────────────────────────

PALETTE = {
    "deepseek-v4":         "#EF5350",
    "qwen3-max-thinking":  "#AB47BC",
    "gpt-5-mini":          "#42A5F5",
    "claude-opus":         "#FF7043",
    "gemini-3-flash":      "#26A69A",
    "gpt-5":               "#78909C",
}

REASONING_MODELS = {
    "deepseek-v4", "qwen3-max-thinking", "gpt-5-mini", "claude-opus", "gemini-3-flash"
}

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#e6edf3",
    "ytick.color": "#e6edf3",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "figure.dpi": 150,
})


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_results(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("results_*.json"))
        if not files:
            raise FileNotFoundError(f"No results_*.json files found in {path}")
        p = files[-1]
        print(f"Using latest results file: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["detected_impossible"] = df["detected_impossible"].astype(bool)
    df["abstained"] = df["abstained"].astype(bool)
    df["correct"] = df["detected_impossible"]  # alias for readability
    return df


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("model").agg(
        detection_rate=("detected_impossible", "mean"),
        abstain_rate=("abstained", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        n=("question_id", "count"),
    ).reset_index()
    grp["detection_pct"] = grp["detection_rate"] * 100
    grp["is_reasoning"] = grp["model"].isin(REASONING_MODELS)
    grp = grp.sort_values("detection_pct", ascending=False)
    return grp


def model_lang_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["model", "language"]).agg(
        detection_rate=("detected_impossible", "mean"),
    ).reset_index()


def model_category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="model",
        columns="category",
        values="detected_impossible",
        aggfunc="mean",
    )
    return pivot * 100  # to percent


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_detection_rate(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [PALETTE.get(m, "#888") for m in summary["model"]]
    bars = ax.barh(summary["model"], summary["detection_pct"], color=colors, height=0.6)

    # Value labels
    for bar, val in zip(bars, summary["detection_pct"]):
        ax.text(
            min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left", fontsize=9,
        )

    # Reasoning vs non-reasoning annotation
    for i, row in summary.reset_index().iterrows():
        tag = "🧠 reasoning" if row["is_reasoning"] else "💬 standard"
        ax.text(
            -1, i, tag,
            va="center", ha="right", fontsize=7, color="#8b949e",
        )

    ax.set_xlabel("Detection Rate (%)")
    ax.set_title("Who Recognizes the Impossible? — Missing Premise Detection Rate", pad=12)
    ax.set_xlim(-18, 110)
    ax.axvline(50, color="#f0883e", linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="x")
    ax.invert_yaxis()

    fig.tight_layout()
    path = out_dir / "01_detection_rate.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_token_usage(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [PALETTE.get(m, "#888") for m in summary["model"]]
    bars = ax.barh(summary["model"], summary["avg_output_tokens"], color=colors, height=0.6)

    for bar, val in zip(bars, summary["avg_output_tokens"]):
        ax.text(
            val + 5, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Average Output Tokens per Question")
    ax.set_title("Token Cost of Overconfidence — Avg Output Tokens per Question", pad=12)
    ax.grid(axis="x")
    ax.invert_yaxis()

    fig.tight_layout()
    path = out_dir / "02_token_usage.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_efficiency_frontier(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in summary.iterrows():
        color = PALETTE.get(row["model"], "#888")
        marker = "D" if row["is_reasoning"] else "o"
        ax.scatter(
            row["avg_output_tokens"],
            row["detection_pct"],
            color=color,
            marker=marker,
            s=120,
            zorder=3,
        )
        ax.annotate(
            row["model"],
            (row["avg_output_tokens"], row["detection_pct"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Avg Output Tokens (lower = cheaper)")
    ax.set_ylabel("Detection Rate % (higher = smarter)")
    ax.set_title("Efficiency Frontier: Detection Rate vs Token Cost", pad=12)
    ax.grid(True)

    legend_elements = [
        mpatches.Patch(color="#aaa", label="◆ Reasoning model"),
        mpatches.Patch(color="#555", label="● Standard model"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    # Ideal quadrant annotation
    ax.axhline(50, color="#f0883e", linestyle="--", alpha=0.3)
    ax.text(ax.get_xlim()[0] + 5, 52, "↑ Better detection", fontsize=8, color="#f0883e", alpha=0.6)

    fig.tight_layout()
    path = out_dir / "03_efficiency_frontier.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_category_heatmap(matrix: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Detection Rate (%)")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, fontsize=10)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            color = "black" if val > 55 else "white"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=10, color=color)

    ax.set_title("Detection Rate by Model × Category", pad=12)
    fig.tight_layout()
    path = out_dir / "04_category_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_language_comparison(lang_df: pd.DataFrame, out_dir: Path) -> None:
    pivot = lang_df.pivot(index="model", columns="language", values="detection_rate") * 100
    pivot = pivot.sort_values("es", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot))
    width = 0.35

    bars_es = ax.bar(x - width / 2, pivot["es"], width, label="Español", color="#58a6ff", alpha=0.85)
    bars_en = ax.bar(x + width / 2, pivot["en"], width, label="English", color="#3fb950", alpha=0.85)

    for bars in (bars_es, bars_en):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Spanish vs English — Does Language Affect Reasoning?", pad=12)
    ax.legend()
    ax.grid(axis="y")
    ax.set_ylim(0, 115)

    fig.tight_layout()
    path = out_dir / "05_language_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def save_summary_table(summary: pd.DataFrame, df: pd.DataFrame, out_dir: Path) -> None:
    # Cost per correct detection
    correct_per_model = df[df["detected_impossible"]].groupby("model").size().rename("n_correct")
    merged = summary.merge(correct_per_model, on="model", how="left").fillna(0)
    merged["cost_per_correct"] = merged.apply(
        lambda r: r["total_cost_usd"] / r["n_correct"] if r["n_correct"] > 0 else float("inf"),
        axis=1,
    )

    cols = [
        "model", "detection_pct", "abstain_rate", "avg_output_tokens",
        "total_cost_usd", "cost_per_correct", "is_reasoning",
    ]
    out = merged[cols].copy()
    out.columns = [
        "Model", "Detection %", "Abstain Rate", "Avg Output Tokens",
        "Total Cost USD", "Cost per Correct", "Is Reasoning",
    ]
    out["Detection %"] = out["Detection %"].round(1)
    out["Abstain Rate"] = (out["Abstain Rate"] * 100).round(1)
    out["Avg Output Tokens"] = out["Avg Output Tokens"].round(0).astype(int)
    out["Total Cost USD"] = out["Total Cost USD"].round(4)
    out["Cost per Correct"] = out["Cost per Correct"].round(4)

    path = out_dir / "summary_table.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved summary table: {path}")
    print("\n" + out.to_string(index=False))


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Missing Premise Benchmark — Analysis")
    parser.add_argument(
        "--results",
        default="results/",
        help="Path to results JSON file or directory (uses latest if directory)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/",
        help="Directory to save plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    df = load_results(args.results)
    print(f"Loaded {len(df)} rows from {len(df['model'].unique())} models, {len(df['question_id'].unique())} questions\n")

    summary = model_summary(df)
    lang_df = model_lang_summary(df)
    cat_matrix = model_category_matrix(df)

    plot_detection_rate(summary, out_dir)
    plot_token_usage(summary, out_dir)
    plot_efficiency_frontier(summary, out_dir)
    plot_category_heatmap(cat_matrix, out_dir)
    plot_language_comparison(lang_df, out_dir)
    save_summary_table(summary, df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
