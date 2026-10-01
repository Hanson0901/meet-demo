#!/usr/bin/env python3
"""Benchmark generation speed at each successful n_ctx from the limit run."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_MODEL = "/home/cgu-csie/qwen3-4b-instruct-2507-q8_0.gguf"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mem_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields = value.split()
        if fields:
            values[key] = int(fields[0]) * (1024 if fields[-1] == "kB" else 1)
    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
            key, value = line.split(":", 1)
            fields = value.split()
            if fields:
                status[key] = int(fields[0]) * (1024 if fields[-1] == "kB" else 1)
    return {
        "rss_bytes": status.get("VmRSS", 0),
        "hwm_bytes": status.get("VmHWM", 0),
        "vms_bytes": status.get("VmSize", 0),
        "mem_available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
        "swap_used_bytes": values.get("SwapTotal", 0) - values.get("SwapFree", 0),
    }


def emit(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def trial(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    before = mem_snapshot()
    emit({"kind": "speed_start", "time": now(), "context": args.context, "before": before, "pid": os.getpid()})
    llm = None
    try:
        from llama_cpp import Llama

        threads = args.threads or max(1, (os.cpu_count() or 4) - 1)
        llm = Llama(
            model_path=args.model,
            n_ctx=args.context,
            n_batch=min(512, args.context),
            n_ubatch=min(512, args.context),
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            flash_attn=False,
            verbose=False,
        )

        # Warm up the execution graph, then time only generation.  The same
        # one-token prompt is used for every context, so the table compares
        # the effect of n_ctx and its KV allocation rather than prompt size.
        bos = llm.token_bos()
        llm.eval([bos])
        generation_start = time.perf_counter()
        generated = []
        for _ in range(args.tokens):
            token = llm.sample(top_k=1, top_p=1.0, temperature=0.0, repeat_penalty=1.0)
            generated.append(int(token))
            llm.eval([token])
        generation_seconds = time.perf_counter() - generation_start
        after = mem_snapshot()
        emit({
            "kind": "speed_result",
            "time": now(),
            "context": args.context,
            "ok": True,
            "generated_tokens": len(generated),
            "generation_seconds": round(generation_seconds, 6),
            "generation_tps": round(len(generated) / generation_seconds, 6),
            "runtime_context": llm.n_ctx(),
            "after": after,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        return 0
    except BaseException as exc:
        emit({
            "kind": "speed_result",
            "time": now(),
            "context": args.context,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "after": mem_snapshot(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        return 2
    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        gc.collect()


def successful_contexts(path: Path) -> list[int]:
    contexts: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "trial_result" and record.get("ok") is True:
            contexts.add(int(record["context"]))
    return sorted(contexts)


def run_one(args: argparse.Namespace, context: int, log) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--trial",
        str(context),
        "--model",
        args.model,
        "--threads",
        str(args.threads or 0),
        "--tokens",
        str(args.tokens),
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        output, _ = proc.communicate()
    log.write(output)
    log.flush()
    print(output, end="")
    records = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "speed_result":
            records.append(record)
    child = records[-1] if records else {}
    result = {
        "kind": "speed_driver_result",
        "time": now(),
        "context": context,
        "ok": proc.returncode == 0 and child.get("ok") is True,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "generated_tokens": child.get("generated_tokens", 0),
        "generation_seconds": child.get("generation_seconds", 0),
        "generation_tps": child.get("generation_tps", 0),
        "hwm_bytes": child.get("after", {}).get("hwm_bytes", 0),
        "swap_used_bytes": child.get("after", {}).get("swap_used_bytes", 0),
        "error": child.get("error", "") if not child.get("ok") else "",
    }
    emit(result)
    return result


def driver(args: argparse.Namespace) -> int:
    contexts = successful_contexts(Path(args.limit_results))
    if not contexts:
        emit({"kind": "speed_error", "time": now(), "error": "no successful contexts in limit results"})
        return 1
    result_path = Path(args.results)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with result_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "speed_run_start", "time": now(), "args": vars(args), "contexts": contexts}, ensure_ascii=False) + "\n")
        log.flush()
        emit({"kind": "speed_run_start", "time": now(), "contexts": contexts, "results": str(result_path)})
        for context in contexts:
            row = run_one(args, context, log)
            rows.append(row)

        summary = {
            "kind": "speed_run_summary",
            "time": now(),
            "model": args.model,
            "tokens_per_trial": args.tokens,
            "rows": rows,
        }
        log.write(json.dumps(summary, ensure_ascii=False) + "\n")
        log.flush()
        emit(summary)
    return 0 if any(row["ok"] for row in rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trial", type=int, metavar="CONTEXT")
    mode.add_argument("--driver", action="store_true")
    parser.add_argument("--model", default=os.environ.get("LLAMA_CPP_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit-results", default="data/llama-context-limit-20261001.jsonl")
    parser.add_argument("--results", default="data/llama-context-speed-20261001.jsonl")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trial is not None:
        args.context = args.trial
        return trial(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
