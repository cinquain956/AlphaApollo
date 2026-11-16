# Task 2: SkyDiscover search comparison

Task 2 connects the deterministic Task 1 geometry evaluators to the pip-installed
`skydiscover` package. No SkyDiscover source is copied into this repository.
The configurable algorithms are `openevolve_native` and `evox`.

## Pipeline

```text
SkyDiscover search algorithm
  -> Qwen3.8-27B OpenAI-compatible API call
  -> complete parameterised Python candidate
  -> Task 1 subprocess executor
  -> deterministic Circle Packing or MMD checker
  -> score + detailed violation feedback
  -> SkyDiscover population / parent selection
```

`openevolve_native` and `evox` share the evaluator, initial program, model, number
of solution iterations, and output schema. Only the SkyDiscover search type and
its database configuration change.

## Installation

```bash
pip install "skydiscover>=0.2.0"
pip install matplotlib pyyaml
```

The repository also exposes these packages through the `discovery` optional
extra.

## API configuration

The checked-in config reads `QWEN_API_BASE` and `QWEN_API_KEY`. Set them to an
OpenAI-compatible Bailian or OpenRouter endpoint and key. The default `model.name` is Bailian's `qwen3.8-27b`. For OpenRouter, set it
to `qwen/qwen3.8-27b`. Credentials are never written to run artifacts.

PowerShell example:

```powershell
$env:QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_API_KEY = "..."
python demo_t2.py run
```

Validate the complete 12-run matrix and both initial programs without making an
API request:

```bash
python demo_t2.py dry-run
```

## Experiment design

The default configuration runs:

- 2 algorithms: `openevolve_native`, `evox`
- 2 problems: `circle_packing`, `mmd`
- 3 seeds: `11`, `22`, `33`
- 32 solution iterations per run

Runs from different seeds execute concurrently with `max_workers: 3`. Each run
has a separate seeded population and output directory.

## Persistence

Every run writes:

- `programs.jsonl`: every retained executable candidate, full source, score,
  metrics, parent, generation, and iteration
- `best_so_far.csv`: iterations `0..32`, including carry-forward after a failed
  generation round
- `llm_calls.jsonl`: one row per real API attempt, including retries
- `run_config.json`: reproducibility metadata without secrets
- native SkyDiscover checkpoints and best-program files

LLM calls are classified as:

- `solution`: may generate a geometry candidate
- `strategy`: EvoX search-policy or variation-operator generation; no geometry
  candidate is produced
- `probe`: endpoint availability checks

This separation prevents EvoX's additional meta-search compute from being
mistaken for candidate count.

## Analysis outputs

`python demo_t2.py analyze` creates:

- four independent plots under `outputs/t2/plots/`
- three-seed mean with a shaded population standard deviation
- identical x-axis limit of 32 iterations
- `raw_best_so_far.csv` with every seed's original curve
- `llm_call_counts.csv`
- `aggregate.json` and `report.md`

The report only claims an evolving effect when the three-seed mean final
best-so-far exceeds its initial mean. Actual scientific conclusions require all
12 API runs; dry-run results are validation data, not experimental results.
