"""Summarize prefill steps and checkpoint kernels from torch profiler traces."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
import gzip
import json
from pathlib import Path
import re


RECURRENT_CHECKPOINT_KERNELS = (
    "_store_recurrent_checkpoint",
)
CHECKPOINT_KERNELS = ("_store_conv_checkpoint", *RECURRENT_CHECKPOINT_KERNELS)


def analyze(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        events = json.load(handle)["traceEvents"]
    windows = []
    launches = {}
    kernels = []
    for event in events:
        category = event.get("cat")
        if category == "user_annotation":
            name = event.get("name", "")
            match = re.search(r"execute_context_\d+\((\d+)\)", name)
            if match:
                windows.append((event["ts"], event["ts"] + event["dur"], int(match[1])))
        elif category in ("cuda_runtime", "hip_runtime"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = event["ts"]
        elif category == "kernel":
            kernels.append(event)
    windows.sort()
    starts = [window[0] for window in windows]
    steps = [
        {"tokens": window[2], "cpu_ms": (window[1] - window[0]) / 1000,
         "gpu_sum_ms": 0.0, "nccl_ms": 0.0, "checkpoint_kernels": Counter()}
        for window in windows
    ]
    counts = Counter()
    unmatched = 0
    for event in kernels:
        name = event["name"]
        checkpoint_name = next((key for key in CHECKPOINT_KERNELS if key in name), None)
        if checkpoint_name:
            counts[checkpoint_name] += 1
        timestamp = launches.get(event.get("args", {}).get("correlation"))
        if timestamp is None:
            unmatched += 1
            continue
        index = bisect.bisect_right(starts, timestamp) - 1
        if index < 0 or timestamp >= windows[index][1]:
            continue
        steps[index]["gpu_sum_ms"] += event["dur"] / 1000
        if "nccl" in name.lower():
            steps[index]["nccl_ms"] += event["dur"] / 1000
        if checkpoint_name:
            steps[index]["checkpoint_kernels"][checkpoint_name] += 1
    active_steps = [step for step in steps if step["tokens"]]
    tokens = sum(step["tokens"] for step in active_steps)
    normalized = {
        key.replace("_ms", "_us_per_token"): sum(step[key] for step in active_steps) * 1000 / tokens
        for key in ("gpu_sum_ms", "nccl_ms")
    } if tokens else {}
    return {"path": str(path), "checkpoint_kernels": dict(counts), "steps": steps,
            "normalized": normalized,
            "unmatched_kernels": unmatched, "total_kernels": len(kernels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args()
    results = [analyze(path) for path in args.traces]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(Path(result["path"]).name, result["checkpoint_kernels"])
        print("step tokens:", [step["tokens"] for step in result["steps"]])
        print("normalized:", result["normalized"])
    if args.require_checkpoint and not any(
        result["checkpoint_kernels"].get("_store_conv_checkpoint", 0)
        and any(result["checkpoint_kernels"].get(key, 0) for key in RECURRENT_CHECKPOINT_KERNELS)
        for result in results
    ):
        raise SystemExit("No trace contains both conv and recurrent checkpoint exports")


if __name__ == "__main__":
    main()
