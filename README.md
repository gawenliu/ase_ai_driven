# ASE AI-Driven Candidate Search

This repository contains a course-project prototype built on top of
[A.S.E / AICGSecEval](https://github.com/Tencent/AICGSecEval), a repository-level
benchmark for evaluating security in AI-generated code.

The original A.S.E framework provides the evaluator. This project adds an
AI-driven candidate-search layer around it:

- a fixed `evaluate(candidate)` wrapper;
- candidate JSON configurations for retrieval, prompting, and generation knobs;
- batch runners for many candidates and repeats;
- AI-guided candidate generation and prompt-iterative optimization;
- analysis scripts and reduced-scale result artifacts for the final case study.

## Project Structure

```text
.
├── invoke.py                         # upstream A.S.E entrypoint with minimal LLM-mode wiring
├── run_code_generation_llm.py         # upstream generation path with candidate prompt knobs
├── run_security_scan.py               # upstream security scan
├── run_evaluate.py                    # upstream A.S.E scoring
├── bench/                             # upstream A.S.E helper modules
├── scripts/
│   ├── make_tiny_dataset.py           # create small task subsets
│   ├── run_eval.py                    # fixed evaluate(candidate) wrapper
│   ├── run_many.py                    # run many candidate JSONs
│   ├── run_ai_search.py               # AI proposes candidate configs from prior results
│   ├── run_prompt_iterative.py        # feedback-driven prompt/template optimizer
│   ├── parse_results.py               # normalize A.S.E artifacts into summary JSON
│   ├── analyze_results.py             # report snippets and plots
│   └── compute_refined_metrics.py     # optional auxiliary continuous metrics
├── configs/
│   ├── candidates/                    # seed baseline candidates
│   └── candidates_final/              # final experiment candidate sets
├── data/
│   ├── data_v2.json                   # reduced copy of upstream metadata
│   ├── data_v2_tiny.json              # 1-task demo subset
│   ├── data_v2_n5.json                # 5-task subset
│   ├── data_v2_n10.json               # 10-task subset
│   └── data_v2_context_bm25.jsonl     # retrieval context used by A.S.E
├── results/analysis_compare/          # compact final comparison CSVs and figures
├── docs/PROTO_EVAL.md                 # implementation notes
└── case_study/                        # one-page case study source and figures
```

## Requirements

Recommended system environment follows A.S.E:

- Python 3.11+
- Docker 27+
- RAM 16GB+
- Disk 100GB+ if running Docker-based security scans on multiple tasks

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check Docker:

```bash
docker --version
docker ps
```

If `docker ps` fails due to permissions, add the user to the `docker` group or
run with an environment where Docker is accessible.

## API Configuration

The LLM generation path needs an OpenAI-compatible API endpoint. Do not commit
real secrets. Create a local `config.ini` from the example:

```bash
cp config.example.ini config.ini
```

Then edit:

```ini
[default]
model_name = deepseek-chat
base_url = https://api.deepseek.com/v1
api_key = YOUR_LLM_API_KEY
github_token = YOUR_GITHUB_TOKEN
```

`github_token` is optional but recommended because A.S.E clones public GitHub
repositories and anonymous cloning can hit rate limits.

You can also provide values through environment variables:

```bash
export MODEL_NAME=deepseek-chat
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export API_KEY=...
export GITHUB_TOKEN=...
```

## Small Demo

Create a one-task dataset:

```bash
python3 scripts/make_tiny_dataset.py \
  --src ./data/data_v2.json \
  --out ./data/data_v2_demo.json \
  --n 1
```

Run one candidate through the fixed evaluator:

```bash
python3 scripts/run_eval.py \
  --candidate configs/candidates/v0.json \
  --dataset ./data/data_v2_demo.json \
  --output_root ./outputs_demo \
  --config ./config.ini \
  --num_cycles 1 \
  --max_workers 1
```

Expected outputs:

```text
outputs_demo/
├── <candidate>_r1_<timestamp>/
│   ├── candidate.json
│   ├── invoke_stdout.log
│   ├── invoke_stderr.log
│   ├── run.log
│   └── summary.json
└── results.csv
```

The evaluator records failed runs as structured summaries rather than dropping
them. This is important because failures are part of the candidate quality
signal.

## Batch Baseline Sweep

Run the seed candidate set on the one-task demo:

```bash
python3 scripts/run_many.py \
  --candidates_dir configs/candidates \
  --dataset ./data/data_v2_tiny.json \
  --output_root ./outputs_demo/baseline \
  --repeat 1 \
  --max_workers 1 \
  --parallel_candidates 1 \
  --config ./config.ini \
  --num_cycles 1
```

Analyze the resulting table:

```bash
python3 scripts/analyze_results.py \
  --input ./outputs_demo/baseline/summary_all.csv \
  --outdir ./outputs_demo/baseline
```

This writes `REPORT_SNIPPET.md` plus plots under the output directory.

## AI-Driven Search

General candidate search, where the AI proposes retrieval/prompt/generation
configurations:

```bash
python3 scripts/run_ai_search.py \
  --dataset ./data/data_v2_n5.json \
  --output_root ./outputs_demo/ai_search \
  --seed_candidates_dir ./configs/candidates \
  --ai_candidates_root ./configs/demo_ai_candidates \
  --rounds 2 \
  --per_round 2 \
  --repeat 1 \
  --max_workers 1 \
  --parallel_candidates 1 \
  --config ./config.ini \
  --num_cycles 1
```

Prompt-iterative optimization, where the prompt/template settings are revised
from previous-round feedback while other settings stay closer to a baseline:

```bash
python3 scripts/run_prompt_iterative.py \
  --dataset ./data/data_v2_n5.json \
  --output_root ./outputs_demo/prompt_iter \
  --candidates_root ./configs/demo_prompt_iter \
  --rounds 2 \
  --per_round 2 \
  --parallel_candidates 1 \
  --num_cycles 1 \
  --config ./config.ini
```

For the final experiment I used 4-way comparison groups:

- AI General Search
- AI Prompt-Iterative
- Random Baseline
- Static Baseline

The compact final comparison artifacts are included in
`results/analysis_compare/`.

## Metrics

Headline results use official A.S.E metrics only:

- `overall_score`
- `code_quality_score`
- `code_security_score`
- build/static check success flags
- PoC check success flags

This project also computes optional auxiliary continuous metrics, such as
`refined`, `combined`, and `contrast`, for search diagnostics. These are not
official A.S.E scores and should not be mixed into the headline benchmark table.

## Key Finding Snapshot

The final reduced-scale comparison in `results/analysis_compare/compare_4way.csv`
contains 16 runs per scheme:

| Scheme | Official overall mean | Std | Best |
|---|---:|---:|---:|
| AI General Search | 30.22 | 1.14 | 34.08 |
| AI Prompt-Iterative | 30.10 | 1.46 | 34.08 |
| Random Baseline | 28.97 | 2.43 | 30.10 |
| Static Baseline | 27.97 | 4.17 | 30.10 |

The defensible interpretation is modest: AI-guided candidate generation improves
the candidate distribution under a limited run budget. It does not show a smooth
monotonic learning curve, but it reaches a better best observed score and lower
variance than the static baseline.

## Reproducibility Notes

- `outputs_*`, logs, cloned repositories, generated code, and `config.ini` are
  intentionally ignored by Git.
- Re-running the same A.S.E command can recover from checkpoints.
- LLM outputs are stochastic unless the backend fully respects deterministic
  sampling parameters.
- Full A.S.E scans require Docker images referenced by the dataset metadata.

## Provenance

This repository includes a lightly wrapped copy of AICGSecEval components needed
for the course project. The original upstream framework is available at:

https://github.com/Tencent/AICGSecEval
