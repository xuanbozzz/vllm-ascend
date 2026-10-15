# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated native probe processes, deadlines and live terminal output."""

import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import regex as re

from tools.vq2a8_live_log import LiveChildLog

REPO = Path(__file__).resolve().parents[1]
PREFIX = "STARTUP_PROBE "


def child_environment(args, environ=None):
    result = dict(os.environ if environ is None else environ)
    for key in (
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "DEVICE_ID",
        "RANK_ID",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        result.pop(key, None)
    result["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_npu)
    result["ASCEND_LAUNCH_BLOCKING"] = args.launch_blocking
    result["PYTHONUNBUFFERED"] = "1"
    result["PYTHONPATH"] = str(REPO) + (os.pathsep + result["PYTHONPATH"] if result.get("PYTHONPATH") else "")
    return result


def emit(case, event, **fields):
    print(PREFIX + json.dumps({"case": case, "event": event, **fields}, ensure_ascii=False), flush=True)


def stage_recorder(case, synchronize):
    @contextmanager
    def stage(name):
        started = time.monotonic()
        emit(case, "BEGIN", stage=name)
        try:
            yield
            # Distinguish an op/API that does not return from a subsequent wait.
            emit(case, "SUBMITTED", stage=name, elapsed_s=round(time.monotonic() - started, 6))
            synchronize()
        except BaseException as exc:
            emit(case, "FAIL", stage=name, error=str(exc), elapsed_s=round(time.monotonic() - started, 6))
            raise
        emit(case, "PASS", stage=name, elapsed_s=round(time.monotonic() - started, 6))

    return stage


def read_events(log):
    events = []
    with Path(log).open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith(PREFIX):
                try:
                    record = json.loads(line[len(PREFIX) :])
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    events.append(record)
    return events


def parse_snapshot(text, physical_npu):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "NPU ID" in line and "Process id" in line), None)
    if start is None:
        return "unknown"
    table = "\n".join(lines[start:])
    idle = {int(x) for x in re.findall(r"No running processes found in NPU\s+(\d+)\s*\|", table)}
    busy = {int(x) for x in re.findall(r"(?m)^\s*\|\s*(\d+)\s*\|\s*\d+\s*\|", table)}
    if physical_npu in idle and physical_npu not in busy:
        return "idle"
    if physical_npu in busy and physical_npu not in idle:
        return "busy"
    return "unknown"


def terminate_child(process):
    """Bounded cleanup of our child/session only; no device reset or broad pkill."""
    if process.poll() is not None:
        return True
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=3)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            return process.poll() is not None
    except ProcessLookupError:
        return process.poll() is not None
    return True


def run_child(command, environment, log, timeout_s):
    started = time.monotonic()
    timed_out, reaped = False, True
    with Path(log).open("x", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            env=environment,
            cwd=REPO,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        with LiveChildLog(Path(log), "tp1-startup"):
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                reaped = terminate_child(process)
            except BaseException:
                terminate_child(process)
                raise
    events = read_events(log)
    completed = any(event.get("event") == "CASE_PASS" for event in events)
    status = "TIMEOUT" if timed_out else "PASS" if process.returncode == 0 and completed else "FAIL"
    return {
        "status": status,
        "exit_code": process.returncode,
        "reaped": reaped,
        "elapsed_s": round(time.monotonic() - started, 3),
        "log": str(log),
        "last_event": events[-1] if events else None,
        "events": events,
    }
