#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _scan_existing_summaries(output_root: str) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(output_root, "*", "summary.json")):
        try:
            summaries.append(_read_json(path))
        except Exception:
            continue
    return summaries


def _success_index(summaries: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for item in summaries:
        if item.get("status") != "success":
            continue
        cid = str(item.get("candidate_id") or "")
        ridx = int(item.get("repeat_index") or 1)
        dataset = os.path.abspath(str(item.get("dataset_path") or ""))
        if not cid or not dataset:
            continue
        idx[(cid, ridx, dataset)] = item
    return idx


def _infer_failure_reason(summary: Dict[str, Any]) -> Optional[str]:
    if summary.get("status") == "success":
        return None
    run_log = summary.get("paths", {}).get("run_log")
    if not run_log or not os.path.exists(run_log):
        return "unknown"
    try:
        text = open(run_log, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return "unknown"

    checks = [
        ("llm_not_found_or_404", ["NotFoundError", "404 Not Found", "v1chat/completions"]),
        ("llm_auth_or_connectivity", ["401", "403", "Connection", "timeout", "APIConnectionError"]),
        ("patch_apply_failure", ["模型生成补丁为空", "修复后应用补丁仍然失败", "git apply"]),
        ("scan_missing_or_failed", ["scan_results", "安全扫描失败", "scan_error", "FileNotFoundError"]),
        ("dependency_missing", ["ModuleNotFoundError", "No module named"]),
    ]
    for reason, pats in checks:
        if any(p in text for p in pats):
            return reason
    return "other"


def _flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    quality = summary.get("quality", {})
    security = summary.get("security", {})
    patch = quality.get("patch_applied_success", {})
    checks = quality.get("build_or_static_check_success", {})
    eff = summary.get("candidate_effective", {})
    paths = summary.get("paths", {})

    row = {
        "candidate_id": summary.get("candidate_id"),
        "repeat_index": summary.get("repeat_index"),
        "batch_id": summary.get("batch_id"),
        "status": summary.get("status"),
        "dataset_path": summary.get("dataset_path"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "overall_score": summary.get("overall_score"),
        "quality_score": quality.get("code_quality_score"),
        "security_score": security.get("code_security_score"),
        "stability": summary.get("stability"),
        "patch_success_rate": patch.get("rate"),
        "image_status_check_rate": checks.get("image_status_check_rate"),
        "test_case_check_rate": checks.get("test_case_check_rate"),
        "poc_check_success_rate": security.get("poc_check_success_rate"),
        "retrieval_top_k": eff.get("retrieval_top_k"),
        "include_readme": eff.get("include_readme"),
        "prompt_template": eff.get("prompt_template"),
        "temperature": eff.get("temperature"),
        "max_gen_token": eff.get("max_gen_token"),
        "top_k_applied": summary.get("knob_application", {}).get("retrieval.top_k", {}).get("applied"),
        "include_readme_applied": summary.get("knob_application", {}).get("retrieval.include_readme", {}).get("applied"),
        "template_applied": summary.get("knob_application", {}).get("prompt.template", {}).get("applied"),
        "temperature_applied": summary.get("knob_application", {}).get("generation.temperature", {}).get("applied"),
        "max_tokens_applied": summary.get("knob_application", {}).get("generation.max_tokens", {}).get("applied"),
        "failure_reason": _infer_failure_reason(summary),
        "summary_json": paths.get("summary_json"),
        "run_log": paths.get("run_log"),
    }
    return row


def _write_summary_all(output_root: str, summaries: List[Dict[str, Any]]) -> None:
    os.makedirs(output_root, exist_ok=True)
    summary_json_path = os.path.join(output_root, "summary_all.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    rows = [_flatten_summary(s) for s in summaries]
    summary_csv_path = os.path.join(output_root, "summary_all.csv")
    headers = [
        "candidate_id",
        "repeat_index",
        "batch_id",
        "status",
        "dataset_path",
        "runtime_seconds",
        "overall_score",
        "quality_score",
        "security_score",
        "stability",
        "patch_success_rate",
        "image_status_check_rate",
        "test_case_check_rate",
        "poc_check_success_rate",
        "retrieval_top_k",
        "include_readme",
        "prompt_template",
        "temperature",
        "max_gen_token",
        "top_k_applied",
        "include_readme_applied",
        "template_applied",
        "temperature_applied",
        "max_tokens_applied",
        "failure_reason",
        "summary_json",
        "run_log",
    ]
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_one_candidate(
    candidate_file: str,
    candidate_id: str,
    repeat_idx: int,
    dataset: str,
    output_root: str,
    max_workers: int,
    config: str,
    num_cycles: Optional[int],
    model_name: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    github_token: Optional[str],
) -> Dict[str, Any]:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_id = f"{_safe_name(candidate_id)}_r{repeat_idx}_{timestamp}"
    cmd = [
        "python3",
        os.path.join("scripts", "run_eval.py"),
        "--candidate",
        candidate_file,
        "--dataset",
        dataset,
        "--output_root",
        output_root,
        "--batch_id",
        batch_id,
        "--repeat_index",
        str(repeat_idx),
        "--max_workers",
        str(max_workers),
        "--config",
        config,
    ]
    if num_cycles is not None:
        cmd.extend(["--num_cycles", str(num_cycles)])
    if model_name:
        cmd.extend(["--model_name", model_name])
    if base_url:
        cmd.extend(["--base_url", base_url])
    if api_key:
        cmd.extend(["--api_key", api_key])
    if github_token:
        cmd.extend(["--github_token", github_token])

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    summary_path = os.path.join(output_root, batch_id, "summary.json")
    if os.path.exists(summary_path):
        summary = _read_json(summary_path)
    else:
        summary = {
            "candidate_id": candidate_id,
            "repeat_index": repeat_idx,
            "batch_id": batch_id,
            "dataset_path": dataset,
            "status": "failed",
            "runtime_seconds": None,
            "paths": {"summary_json": summary_path, "run_log": os.path.join(output_root, batch_id, "run.log")},
        }
    summary["run_many_returncode"] = proc.returncode
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run many candidate configs and consolidate summaries")
    parser.add_argument("--candidates_dir", default="configs/candidates")
    parser.add_argument("--dataset", default="./data/data_v2_tiny.json")
    parser.add_argument("--output_root", default="./outputs_course")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--parallel_candidates", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default="./config.ini")
    parser.add_argument("--num_cycles", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--github_token", default=None)
    args = parser.parse_args()

    candidates_dir = os.path.abspath(args.candidates_dir)
    dataset = os.path.abspath(args.dataset)
    output_root = os.path.abspath(args.output_root)
    candidate_files = sorted(glob.glob(os.path.join(candidates_dir, "*.json")))
    if not candidate_files:
        raise ValueError(f"No candidate JSON files found in {candidates_dir}")

    existing = _scan_existing_summaries(output_root)
    success_idx = _success_index(existing)
    all_summaries: List[Dict[str, Any]] = []
    tasks: List[Tuple[str, str, int]] = []

    for candidate_file in candidate_files:
        candidate = _read_json(candidate_file)
        candidate_id = str(candidate.get("candidate_id") or os.path.splitext(os.path.basename(candidate_file))[0])

        for repeat_idx in range(1, max(args.repeat, 1) + 1):
            key = (candidate_id, repeat_idx, dataset)
            if not args.force and key in success_idx:
                summary = dict(success_idx[key])
                summary["run_many_note"] = "skipped_existing_success"
                all_summaries.append(summary)
                continue
            tasks.append((candidate_file, candidate_id, repeat_idx))

    parallel_candidates = max(1, int(args.parallel_candidates))
    if parallel_candidates == 1:
        for candidate_file, candidate_id, repeat_idx in tasks:
            summary = _run_one_candidate(
                candidate_file=candidate_file,
                candidate_id=candidate_id,
                repeat_idx=repeat_idx,
                dataset=dataset,
                output_root=output_root,
                max_workers=args.max_workers,
                config=args.config,
                num_cycles=args.num_cycles,
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key,
                github_token=args.github_token,
            )
            all_summaries.append(summary)
            if summary.get("status") == "success":
                key = (str(summary.get("candidate_id") or ""), int(summary.get("repeat_index") or 1), dataset)
                success_idx[key] = summary
    else:
        with ThreadPoolExecutor(max_workers=parallel_candidates) as ex:
            futs = [
                ex.submit(
                    _run_one_candidate,
                    candidate_file,
                    candidate_id,
                    repeat_idx,
                    dataset,
                    output_root,
                    args.max_workers,
                    args.config,
                    args.num_cycles,
                    args.model_name,
                    args.base_url,
                    args.api_key,
                    args.github_token,
                )
                for candidate_file, candidate_id, repeat_idx in tasks
            ]
            for fut in as_completed(futs):
                summary = fut.result()
                all_summaries.append(summary)
                if summary.get("status") == "success":
                    key = (str(summary.get("candidate_id") or ""), int(summary.get("repeat_index") or 1), dataset)
                    success_idx[key] = summary

    _write_summary_all(output_root, all_summaries)

    success_count = sum(1 for s in all_summaries if s.get("status") == "success")
    fail_count = len(all_summaries) - success_count
    print(
        json.dumps(
            {
                "status": "ok",
                "candidates": len(candidate_files),
                "runs": len(all_summaries),
                "success": success_count,
                "failed": fail_count,
                "summary_all_csv": os.path.join(output_root, "summary_all.csv"),
                "summary_all_json": os.path.join(output_root, "summary_all.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
