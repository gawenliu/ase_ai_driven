# Prototype Evaluator (Course Project)

This document describes a minimal runnable evaluation wrapper for:
- A.S.E: repository-level benchmark for security in AI-generated code
- Framework: `Tencent/AICGSecEval`

The wrapper keeps AICGSecEval internals unchanged and invokes `invoke.py` through subprocess.

## Prerequisites

- Python `>= 3.11`
- Docker installed and daemon accessible
- GitHub token (recommended to avoid clone/rate-limit issues)
- LLM API credentials (`model_name`, `base_url`, `api_key`)

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## 1) Build Tiny Dataset

Default (take first 1 task):

```bash
python3 scripts/make_tiny_dataset.py --n 1
```

Custom source/output:

```bash
python3 scripts/make_tiny_dataset.py \
  --src ./data/data_v2.json \
  --out ./data/data_v2_tiny.json \
  --n 1
```

Select by task IDs:

```bash
python3 scripts/make_tiny_dataset.py \
  --src ./data/data_v2.json \
  --out ./data/data_v2_tiny.json \
  --n 1 \
  --task_ids Choser_CVE-2016-10504
```

## 2) Run One Evaluation

```bash
python3 scripts/run_eval.py \
  --candidate configs/candidates/v0.json \
  --dataset ./data/data_v2_tiny.json \
  --output_root ./outputs_course \
  --model_name deepseek-chat \
  --base_url https://api.deepseek.com/v1/ \
  --api_key <YOUR_API_KEY> \
  --github_token <YOUR_GITHUB_TOKEN>
```

Notes:
- `--max_workers` defaults to `1` (prototype requirement).
- `--num_cycles` defaults to `3` (framework default behavior).
- `--config ./config.ini` is enabled by default; if you put `api_key/base_url/model_name/github_token` there, you can omit them from CLI.

## 3) Run Many Candidates (Search-Ready Table)

```bash
python3 scripts/run_many.py \
  --candidates_dir configs/candidates \
  --dataset ./data/data_v2_tiny.json \
  --output_root ./outputs_course
```

Optional:
- `--repeat N` to run each candidate multiple times
- `--parallel_candidates M` to run up to `M` candidates concurrently (default `1`)
- `--force` to ignore resumability and rerun successful candidates

Example with candidate-level parallelism:

```bash
python3 scripts/run_many.py \
  --candidates_dir configs/candidates \
  --dataset ./data/data_v2_tiny.json \
  --output_root ./outputs_course \
  --parallel_candidates 2 \
  --max_workers 1
```

For heavy Docker workloads, start with `--parallel_candidates 2` and increase gradually.

Consolidated outputs:
- `outputs_course/summary_all.csv`
- `outputs_course/summary_all.json`

## 4) Analyze Consolidated Results

```bash
python3 scripts/analyze_results.py \
  --input ./outputs_course/summary_all.csv \
  --outdir ./outputs_course
```

Analysis outputs:
- `outputs_course/REPORT_SNIPPET.md`
- `outputs_course/security_vs_quality.png`
- `outputs_course/overall_by_topk.png`

## 5) Parse Existing Output Directory

```bash
python3 scripts/parse_results.py --output_dir ./outputs_course/<batch_id>
```

## Expected Output Layout (one run)

```text
outputs_course/
  <batch_id>/
    candidate.json
    invoke_stdout.log
    invoke_stderr.log
    run.log
    summary.json
    raw_repo/
    generated_code/
      <model_name>__<batch_id>/
        processed_instances.json
        scan_results/
        scan_results.json
        all_metrics.json
      <model_name>__<batch_id>_score.json
  results.csv
  summary_all.csv
  summary_all.json
  REPORT_SNIPPET.md
  security_vs_quality.png
  overall_by_topk.png
```

## Metrics in `summary.json`

- Basic:
  - `candidate_id`, `batch_id`, `dataset_path`, `runtime_seconds`, `status`
- Quality:
  - patch applied success (`processed_instances.success`)
  - build/static-like signals if available (`image_status_check`, `test_case_check`)
- Security:
  - `poc_check` success rate
  - `code_security_score` (if present in framework metrics)
  - `security_alerts_delta` is `null` in current prototype
- Stability:
  - `code_stability_score` if present; otherwise `null`
- Raw artifact paths:
  - `run.log`, `scan_results.json`, `all_metrics.json`, etc.
- Candidate application metadata:
  - `candidate`
  - `candidate_effective`
  - `knob_application` (whether each knob is actually applied)

## Candidate Knobs Applied in Current Version

The following fields now affect runtime behavior:
- `retrieval.top_k` -> limits retrieval hits used to build context
- `retrieval.include_readme` -> controls whether README files are included in prompt context
- `prompt.template` -> one of:
  - `default_unified_diff`
  - `security_minimal_edit`
  - `security_checklist`
- `generation.temperature` -> passed to LLM generation
- `generation.max_tokens` -> mapped to `invoke.py --max_gen_token`

If a field is missing or invalid, it is still recorded and marked as `applied=false` in `summary.json`.

## Troubleshooting

1. Docker permission / daemon issue
- Check:
  - `docker --version`
  - `docker info`
- If using wrong context:
  - `docker context ls`
  - `docker context use default`
- If permission denied on socket:
  - ensure your user can access Docker daemon

2. GitHub rate limit / clone failures
- Provide `--github_token` (or set `github_token` in `config.ini`)
- Re-run the same command; framework supports checkpoint recovery

3. API connectivity / auth failures
- Verify `base_url`, `model_name`, `api_key`
- Check `invoke_stderr.log` and `run.log` in batch output directory

4. Interrupted execution
- Re-run the same `scripts/run_eval.py` command with the same `--batch_id` to leverage framework checkpoint behavior

## What `evaluate(candidate)` Means in This Prototype

`evaluate(candidate_config)` is currently implemented as:
1. Persist/read candidate JSON (knobs tracked for experiment bookkeeping),
2. Run `invoke.py` once on tiny dataset (`--llm`, `--max_workers 1`),
3. Parse artifacts into one machine-readable `summary.json`,
4. Append one-line record into `results.csv`.

Current limitation:
- Knobs are wired through a lightweight adapter layer, but not every possible candidate field is consumed by AICGSecEval internals.

Next iteration:
- Expand candidate-to-runtime mapping and improve stability estimation with repeated runs and stronger statistical analysis.
