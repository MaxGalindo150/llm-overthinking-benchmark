# Missing Premise Benchmark

> **Can LLMs recognize when a question is unanswerable?**
>
> We test 6 models with 20 impossible questions — 10 math, 10 logic/common sense —
> each in Spanish and English, for 40 total prompts per model.

---

## The Research Question

Modern reasoning models spend thousands of tokens solving problems that *cannot be solved*.
A 10-year-old would immediately say *"you didn't give me enough information"*.
Most LLMs hallucinate an answer instead.

This benchmark measures **epistemic humility**: the ability to recognize that a question
lacks a necessary premise and to refuse to answer rather than fabricate.

---

## Methodology

Inspired by the GSM8K benchmark manipulation in [*"GSM-Symbolic: Understanding the Limitations
of Mathematical Reasoning in Large Language Models"* (Apple, 2024)](https://arxiv.org/abs/2410.05229).

### Question Types

| Type | Count | Description |
|------|-------|-------------|
| Math (missing premise) | 10 | Classic word problems with one necessary value removed |
| Logic / Common Sense | 10 | Paradoxes, contradictions, or ambiguous setups |

Each question is provided in both **Spanish** and **English**.

### Scoring

| Outcome | Score | Description |
|---------|-------|-------------|
| Detected impossible | 1.0 | Model correctly identifies the missing/contradictory premise |
| Abstained without detection | 0.5 | Refused to answer but without explaining why |
| Attempted answer | 0.0 | Hallucinated a response to an unanswerable question |

### Models Evaluated

| Model | Provider | Type |
|-------|----------|------|
| DeepSeek-R1 | DeepSeek | Reasoning |
| Qwen3-Thinking (235B) | OpenRouter | Reasoning |
| GPT-o3-mini | OpenAI | Reasoning |
| Claude Sonnet 4.6 | Anthropic | Standard |
| Gemini 2.5 Flash Thinking | Google | Reasoning |
| GPT-4o | OpenAI | Standard |

---

## Reproducing the Experiment

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/missing-premise-benchmark
cd missing-premise-benchmark
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env with your actual keys
```

### 3. Dry run (no API calls)

```bash
python run_experiments.py --dry-run
```

### 4. Run a subset

```bash
# Test only GPT-4o and Claude, Spanish only
python run_experiments.py --models gpt-4o claude-sonnet --languages es
```

### 5. Run the full benchmark

```bash
python run_experiments.py
```

Results are saved to `results/results_YYYYMMDD_HHMMSS.json` and `.csv`.

### 6. Analyze and visualize

```bash
python analyze.py
# or point to a specific file:
python analyze.py --results results/results_20260222_120000.json
```

Plots are saved to `plots/`.

---

## Repository Structure

```
missing-premise-benchmark/
├── data/
│   └── benchmark.json        # 20 questions × 2 languages
├── results/                  # Created at runtime
│   └── results_*.json
├── plots/                    # Created at runtime
│   ├── 01_detection_rate.png
│   ├── 02_token_usage.png
│   ├── 03_efficiency_frontier.png
│   ├── 04_category_heatmap.png
│   └── 05_language_comparison.png
├── run_experiments.py        # API runner
├── analyze.py                # Analysis & visualization
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Findings

*(to be updated after running experiments)*

- **Reasoning models are not necessarily better** at epistemic humility
- Spending more tokens does not correlate with correct detection
- Language (Spanish vs English) may affect performance differently across models
- The "efficiency frontier" reveals which models give the right answer cheaply

---

## Business Implication

If your LLM pipeline answers impossible questions instead of flagging them,
you are paying for **hallucinated tokens** that carry real downstream risk.
The cheapest model that says "I don't know" on time may be more valuable
than the most expensive reasoning model that confidently fabricates.

---

## Citation

If you use this benchmark, please cite:

```
@misc{missing-premise-benchmark-2026,
  title  = {Missing Premise Benchmark: Evaluating Epistemic Humility in LLMs},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/missing-premise-benchmark}
}
```

---

## License

MIT
