#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_rate(count: Optional[int], total: Optional[int]) -> Optional[float]:
    if count is None or total is None or total == 0:
        return None
    return round(count / total, 6)


def _detect_model_dir(output_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    generated_root = os.path.join(output_dir, "generated_code")
    model_dirs: List[str] = []

    if os.path.isdir(generated_root):
        for name in os.listdir(generated_root):
            path = os.path.join(generated_root, name)
            if os.path.isdir(path):
                model_dirs.append(path)

    if not model_dirs:
        # Fallback: scan for processed_instances.json recursively.
        for path in glob.glob(os.path.join(output_dir, "**", "processed_instances.json"), recursive=True):
            model_dirs.append(os.path.dirname(path))

    if not model_dirs:
        return generated_root if os.path.isdir(generated_root) else None, None, None

    model_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    model_dir = model_dirs[0]
    model_batch = os.path.basename(model_dir)
    model_name = None
    batch_id = None
    if "__" in model_batch:
        model_name, batch_id = model_batch.split("__", 1)
    return generated_root if os.path.isdir(generated_root) else None, model_dir, model_name


def _find_score_files(generated_root: Optional[str], model_dir: Optional[str]) -> Dict[str, Optional[str]]:
    all_metrics = os.path.join(model_dir, "all_metrics.json") if model_dir else None
    score_json: Optional[str] = None

    if generated_root and os.path.isdir(generated_root):
        candidates = glob.glob(os.path.join(generated_root, "*_score.json"))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            score_json = candidates[0]

    return {
        "all_metrics": all_metrics if all_metrics and os.path.exists(all_metrics) else None,
        "score_json": score_json if score_json and os.path.exists(score_json) else None,
    }


def _parse_processed(processed_path: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "total": None,
        "success_count": None,
        "success_rate": None,
        "time_seconds_sum": None,
        "any_success": None,
        "all_success": None,
    }
    if not processed_path or not os.path.exists(processed_path):
        return result

    data = _load_json(processed_path)
    if not isinstance(data, dict):
        return result

    total = len(data)
    success_count = 0
    time_sum = 0.0
    for item in data.values():
        if isinstance(item, dict):
            if bool(item.get("success")):
                success_count += 1
            try:
                time_sum += float(item.get("time", 0))
            except Exception:
                pass

    result["total"] = total
    result["success_count"] = success_count
    result["success_rate"] = _safe_rate(success_count, total)
    result["time_seconds_sum"] = round(time_sum, 6)
    result["any_success"] = success_count > 0
    result["all_success"] = (success_count == total) if total > 0 else None
    return result


def _parse_scan(scan_results_path: Optional[str], scan_result_dir: Optional[str]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    if scan_results_path and os.path.exists(scan_results_path):
        data = _load_json(scan_results_path)
        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
    elif scan_result_dir and os.path.isdir(scan_result_dir):
        for path in glob.glob(os.path.join(scan_result_dir, "*_output.json")):
            try:
                items.append(_load_json(path))
            except Exception:
                continue

    total = len(items)
    patch_file_count = sum(1 for x in items if bool(x.get("patch_file")))
    image_ok_count = sum(1 for x in items if bool(x.get("image_status_check")))
    test_ok_count = sum(1 for x in items if bool(x.get("test_case_check")))
    poc_ok_count = sum(1 for x in items if bool(x.get("poc_check")))
    secure_and_quality_count = sum(
        1 for x in items if bool(x.get("poc_check")) and bool(x.get("test_case_check"))
    )

    return {
        "total": total if total > 0 else None,
        "patch_file_success_count": patch_file_count if total > 0 else None,
        "image_status_success_count": image_ok_count if total > 0 else None,
        "test_case_success_count": test_ok_count if total > 0 else None,
        "poc_success_count": poc_ok_count if total > 0 else None,
        "secure_and_quality_success_count": secure_and_quality_count if total > 0 else None,
        "patch_file_success_rate": _safe_rate(patch_file_count, total),
        "image_status_success_rate": _safe_rate(image_ok_count, total),
        "test_case_success_rate": _safe_rate(test_ok_count, total),
        "poc_success_rate": _safe_rate(poc_ok_count, total),
        "secure_and_quality_success_rate": _safe_rate(secure_and_quality_count, total),
    }


def _extract_framework_scores(score_files: Dict[str, Optional[str]]) -> Dict[str, Any]:
    code_quality_score = None
    code_security_score = None
    code_stability_score = None
    overall_score = None

    if score_files["all_metrics"]:
        try:
            metrics = _load_json(score_files["all_metrics"])
            if isinstance(metrics, dict):
                code_quality_score = metrics.get("code_quality_score")
                code_security_score = metrics.get("code_security_score")
                code_stability_score = metrics.get("code_stability_score")
                overall_score = metrics.get("overall_score")
        except Exception:
            pass

    if (code_quality_score is None or code_security_score is None) and score_files["score_json"]:
        try:
            score_data = _load_json(score_files["score_json"])
            if isinstance(score_data, dict) and isinstance(score_data.get("overall"), dict):
                overall = score_data["overall"]
                code_quality_score = code_quality_score if code_quality_score is not None else overall.get("code_quality_score")
                code_security_score = code_security_score if code_security_score is not None else overall.get("code_security_score")
                code_stability_score = code_stability_score if code_stability_score is not None else overall.get("code_stability_score")
                overall_score = overall_score if overall_score is not None else overall.get("overall_score")
        except Exception:
            pass

    return {
        "overall_score": overall_score,
        "code_quality_score": code_quality_score,
        "code_security_score": code_security_score,
        "code_stability_score": code_stability_score,
    }


def build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = os.path.abspath(args.output_dir)
    generated_root, model_dir, detected_model_name = _detect_model_dir(output_dir)

    detected_batch_id = None
    if model_dir and "__" in os.path.basename(model_dir):
        _, detected_batch_id = os.path.basename(model_dir).split("__", 1)

    processed_path = os.path.join(model_dir, "processed_instances.json") if model_dir else None
    scan_results_path = os.path.join(model_dir, "scan_results.json") if model_dir else None
    scan_result_dir = os.path.join(model_dir, "scan_results") if model_dir else None

    processed_stats = _parse_processed(processed_path if processed_path and os.path.exists(processed_path) else None)
    scan_stats = _parse_scan(
        scan_results_path if scan_results_path and os.path.exists(scan_results_path) else None,
        scan_result_dir if scan_result_dir and os.path.isdir(scan_result_dir) else None,
    )
    score_files = _find_score_files(generated_root, model_dir)
    framework_scores = _extract_framework_scores(score_files)

    summary: Dict[str, Any] = {
        "candidate_id": args.candidate_id,
        "batch_id": args.batch_id or detected_batch_id,
        "dataset_path": args.dataset_path,
        "runtime_seconds": args.runtime_seconds,
        "status": args.status,
        "model_name": args.model_name or detected_model_name,
        "quality": {
            "patch_applied_success": {
                "count": processed_stats["success_count"],
                "total": processed_stats["total"],
                "rate": processed_stats["success_rate"],
                "any_success": processed_stats["any_success"],
                "all_success": processed_stats["all_success"],
            },
            "build_or_static_check_success": {
                "image_status_check_rate": scan_stats["image_status_success_rate"],
                "test_case_check_rate": scan_stats["test_case_success_rate"],
            },
            "code_quality_score": framework_scores["code_quality_score"],
        },
        "security": {
            "poc_check_success_rate": scan_stats["poc_success_rate"],
            "secure_and_quality_success_rate": scan_stats["secure_and_quality_success_rate"],
            "code_security_score": framework_scores["code_security_score"],
            "security_alerts_delta": None,
        },
        "stability": framework_scores["code_stability_score"],
        "overall_score": framework_scores["overall_score"],
        "paths": {
            "output_dir": output_dir,
            "generated_code_dir": generated_root,
            "model_output_dir": model_dir,
            "processed_instances_json": processed_path if processed_path and os.path.exists(processed_path) else None,
            "scan_results_json": scan_results_path if scan_results_path and os.path.exists(scan_results_path) else None,
            "scan_results_dir": scan_result_dir if scan_result_dir and os.path.isdir(scan_result_dir) else None,
            "all_metrics_json": score_files["all_metrics"],
            "score_json": score_files["score_json"],
            "run_log": os.path.join(output_dir, "run.log") if os.path.exists(os.path.join(output_dir, "run.log")) else None,
            "invoke_stdout_log": os.path.join(output_dir, "invoke_stdout.log") if os.path.exists(os.path.join(output_dir, "invoke_stdout.log")) else None,
            "invoke_stderr_log": os.path.join(output_dir, "invoke_stderr.log") if os.path.exists(os.path.join(output_dir, "invoke_stderr.log")) else None,
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse AICGSecEval outputs into a compact summary JSON")
    parser.add_argument("--output_dir", required=True, help="Batch output directory used by invoke.py")
    parser.add_argument("--candidate_id", default=None)
    parser.add_argument("--batch_id", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--runtime_seconds", type=float, default=None)
    parser.add_argument("--status", default="unknown")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--summary_out", default=None, help="Optional path to write summary JSON")
    args = parser.parse_args()

    summary = build_summary(args)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)

    if args.summary_out:
        out_path = os.path.abspath(args.summary_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")

    print(rendered)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[parse_results] error: {exc}", file=sys.stderr)
        raise
