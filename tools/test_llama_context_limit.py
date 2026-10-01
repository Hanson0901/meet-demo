#!/usr/bin/env python3
"""Measure the largest n_ctx accepted by the local llama.cpp model.

The driver keeps each llama.cpp context in a fresh child process.  This is
important for a memory-pressure test: a failed allocation or an OOM-killed
child cannot leave an old KV cache in the driver.

The trial deliberately allocates the requested KV cache and decodes one BOS
token.  Filling hundreds of thousands of tokens would measure prompt
throughput as well as capacity and would make the boundary test impractical
on a Raspberry Pi.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


DEFAULT_MODEL = "/home/cgu-csie/qwen3-4b-instruct-2507-q8_0.gguf"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mem_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields = value.split()
            if fields:
                values[key] = int(fields[0]) * (1024 if fields[-1] == "kB" else 1)
    except Exception:
        pass

    status: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                key, value = line.split(":", 1)
                fields = value.split()
                if fields:
                    status[key] = int(fields[0]) * (1024 if fields[-1] == "kB" else 1)
    except Exception:
        pass

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
    """Run one allocation/decode trial in the current process."""
    started = time.monotonic()
    before = mem_snapshot()
    emit({
        "kind": "trial_start",
        "time": now(),
        "context": args.context,
        "before": before,
        "pid": os.getpid(),
    })

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
            verbose=True,
        )

        # This makes the result a usable llama.cpp context, not only a
        # successful constructor call.
        llm.eval([llm.token_bos()])
        after = mem_snapshot()
        emit({
            "kind": "trial_result",
            "time": now(),
            "context": args.context,
            "ok": True,
            "model_context": llm._model.n_ctx_train(),
            "runtime_context": llm.n_ctx(),
            "tokens_evaluated": llm.n_tokens,
            "after": after,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        llm.close()
        llm = None
        gc.collect()
        return 0
    except BaseException as exc:
        after = mem_snapshot()
        emit({
            "kind": "trial_result",
            "time": now(),
            "context": args.context,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "after": after,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        traceback.print_exc()
        return 2
    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        gc.collect()


def metadata_probe(args: argparse.Namespace) -> int:
    from llama_cpp import Llama

    llm = Llama(
        model_path=args.model,
        n_ctx=512,
        n_batch=512,
        n_threads=args.threads or max(1, (os.cpu_count() or 4) - 1),
        n_gpu_layers=0,
        verbose=False,
    )
    try:
        emit({
            "kind": "model_info",
            "time": now(),
            "model": args.model,
            "model_context": llm._model.n_ctx_train(),
            "metadata": {
                key: value
                for key, value in llm.metadata.items()
                if key in {
                    "general.name",
                    "general.file_type",
                    "qwen3.context_length",
                    "qwen3.block_count",
                    "qwen3.embedding_length",
                    "qwen3.attention.head_count_kv",
                    "qwen3.attention.key_length",
                    "qwen3.attention.value_length",
                }
            },
        })
    finally:
        llm.close()
        gc.collect()
    return 0


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
    ]
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        result = {
            "kind": "driver_result",
            "time": now(),
            "context": context,
            "ok": False,
            "returncode": None,
            "error": f"cannot start trial: {exc}",
        }
        emit(result)
        return result

    try:
        output, _ = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        # The process group contains only this test script and its llama.cpp
        # native threads/processes.  Do not signal any unrelated PID.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            output, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
        returncode = proc.returncode
        timed_out = True
    else:
        returncode = proc.returncode
        timed_out = False

    if output:
        log.write(output)
        log.flush()
        print(output, end="")

    trial_records = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "trial_result":
            trial_records.append(record)

    child_record = trial_records[-1] if trial_records else {}
    result = {
        "kind": "driver_result",
        "time": now(),
        "context": context,
        "ok": bool(returncode == 0 and child_record.get("ok") is True),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "error": child_record.get("error", "") if not child_record.get("ok") else "",
        "hwm_bytes": child_record.get("after", {}).get("hwm_bytes", 0),
        "swap_used_bytes": child_record.get("after", {}).get("swap_used_bytes", 0),
    }
    emit(result)
    return result


def driver(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    result_path = Path(args.results) if args.results else root / "data" / "llama-context-limit.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "run_start", "time": now(), "args": vars(args), "pid": os.getpid()}, ensure_ascii=False) + "\n")
        log.flush()
        emit({"kind": "run_start", "time": now(), "results": str(result_path), "model": args.model})

        # Read the model's own limit in a short-lived process before trying
        # large contexts.  The user asked to stop at the model's theoretical
        # limit, not at an arbitrary machine-dependent value.
        info_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--metadata",
            "--model",
            args.model,
            "--threads",
            str(args.threads or 0),
        ]
        info = subprocess.run(info_command, cwd=root, capture_output=True, text=True, timeout=180)
        log.write(info.stdout)
        log.write(info.stderr)
        log.flush()
        print(info.stdout, end="")
        print(info.stderr, end="", file=sys.stderr)
        model_context = args.max_context
        for line in info.stdout.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "model_info":
                model_context = int(record["model_context"])
        if info.returncode != 0:
            emit({"kind": "run_error", "time": now(), "error": "metadata probe failed", "returncode": info.returncode})
            return 1

        tested: dict[int, dict] = {}
        best: int | None = None
        context = args.start
        coarse = True
        attempts = 0

        while attempts < args.max_attempts and context <= model_context:
            if context < 1:
                break
            if context in tested:
                emit({
                    "kind": "boundary_reached",
                    "time": now(),
                    "reason": "requested backoff reached an already-tested context",
                    "best_context": best,
                    "failed_context": context,
                    "model_context": model_context,
                })
                break

            attempts += 1
            result = run_one(args, context, log)
            tested[context] = result
            if result["ok"]:
                best = context if best is None else max(best, context)
                if context >= model_context:
                    emit({"kind": "limit_reached", "time": now(), "best_context": best, "model_context": model_context})
                    break
                context += args.coarse_step if coarse else args.fine_step
            else:
                if coarse:
                    coarse = False
                context -= args.fail_backoff

        else:
            emit({"kind": "run_error", "time": now(), "error": "attempt limit reached", "attempts": attempts})

        summary = {
            "kind": "run_summary",
            "time": now(),
            "model": args.model,
            "model_context": model_context,
            "best_context": best,
            "attempts": attempts,
            "tested_contexts": len(tested),
            "successful_contexts": [ctx for ctx, value in tested.items() if value["ok"]],
            "failed_contexts": [ctx for ctx, value in tested.items() if not value["ok"]],
        }
        log.write(json.dumps(summary, ensure_ascii=False) + "\n")
        log.flush()
        emit(summary)
    return 0 if best is not None else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trial", type=int, metavar="CONTEXT")
    mode.add_argument("--metadata", action="store_true")
    mode.add_argument("--driver", action="store_true")
    parser.add_argument("--model", default=os.environ.get("LLAMA_CPP_MODEL", DEFAULT_MODEL))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--start", type=int, default=8192)
    parser.add_argument("--coarse-step", type=int, default=1024)
    parser.add_argument("--fine-step", type=int, default=128)
    parser.add_argument("--fail-backoff", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trial is not None:
        args.context = args.trial
        return trial(args)
    if args.metadata:
        return metadata_probe(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
