#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _parse_task_ids(raw_ids: List[str]) -> List[str]:
    task_ids: List[str] = []
    for item in raw_ids:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        task_ids.extend(parts)
    # Keep order while removing duplicates
    seen = set()
    deduped: List[str] = []
    for tid in task_ids:
        if tid in seen:
            continue
        seen.add(tid)
        deduped.append(tid)
    return deduped


def _locate_task_list(data: Any) -> Tuple[str, List[Dict[str, Any]]]:
    if isinstance(data, list):
        return "__root__", data
    if not isinstance(data, dict):
        raise ValueError("Input dataset must be a list or dict")

    list_candidates: List[Tuple[str, List[Dict[str, Any]]]] = []
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            list_candidates.append((key, value))

    if not list_candidates:
        raise ValueError("Cannot find task list in input dataset")

    # Prefer a list whose records contain instance_id.
    for key, value in list_candidates:
        if "instance_id" in value[0]:
            return key, value
    return list_candidates[0]


def _pick_tasks(
    tasks: List[Dict[str, Any]],
    n: int,
    task_ids: List[str],
) -> List[Dict[str, Any]]:
    if n <= 0:
        raise ValueError("--n must be > 0")

    if not task_ids:
        if len(tasks) < n:
            raise ValueError(f"Requested n={n}, but dataset has only {len(tasks)} tasks")
        return tasks[:n]

    id_to_task: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        if "instance_id" in task:
            id_to_task[str(task["instance_id"])] = task

    missing = [tid for tid in task_ids if tid not in id_to_task]
    if missing:
        raise ValueError(f"task_ids not found in dataset: {missing}")

    selected = [id_to_task[tid] for tid in task_ids]
    if len(selected) < n:
        raise ValueError(f"Requested n={n}, but only {len(selected)} tasks selected by task_ids")
    return selected[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a tiny dataset from AICGSecEval data file")
    parser.add_argument("--src", default="./data/data_v2.json", help="Source dataset path")
    parser.add_argument("--out", default="./data/data_v2_tiny.json", help="Output tiny dataset path")
    parser.add_argument("--n", type=int, default=1, help="Number of tasks to keep")
    parser.add_argument(
        "--task_ids",
        nargs="*",
        default=[],
        help="Optional task IDs (instance_id). Supports space-separated or comma-separated values",
    )
    args = parser.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)

    container_key, tasks = _locate_task_list(data)
    selected = _pick_tasks(tasks, args.n, _parse_task_ids(args.task_ids))

    if container_key == "__root__":
        tiny_data: Any = selected
    else:
        tiny_data = dict(data)
        tiny_data[container_key] = selected

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tiny_data, f, ensure_ascii=False, indent=2)

    result = {
        "status": "ok",
        "src": os.path.abspath(args.src),
        "out": os.path.abspath(args.out),
        "total_tasks_in_src": len(tasks),
        "selected_tasks": len(selected),
        "selected_instance_ids": [t.get("instance_id") for t in selected],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[make_tiny_dataset] error: {exc}", file=sys.stderr)
        raise
