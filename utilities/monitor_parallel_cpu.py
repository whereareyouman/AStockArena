#!/usr/bin/env python3
"""Sample CPU/RAM while parallel model workers run."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "settings/runtime/cpu_test_10models_2026-06-17.json"
DURATION_SEC = int(os.getenv("CPU_MONITOR_SECONDS", "120"))
SAMPLE_INTERVAL = float(os.getenv("CPU_MONITOR_INTERVAL", "1.0"))


def _project_pythons() -> list[tuple[int, float, float, str]]:
    rows: list[tuple[int, float, float, str]] = []
    proc = subprocess.run(
        ["ps", "-ax", "-o", "pid=,pcpu=,pmem=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    marker = str(PROJECT_ROOT)
    for line in proc.stdout.splitlines():
        if "main.py" not in line or marker not in line:
            continue
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, cpu, mem, cmd = parts
        try:
            rows.append((int(pid), float(cpu), float(mem), cmd))
        except ValueError:
            continue
    return rows


def main() -> int:
    env = os.environ.copy()
    env.update(
        {
            "SKIP_NEWS_PREFETCH": "1",
            "SKIP_AUTO_BACKUP": "1",
            "NEWS_ALREADY_PREFETCHED": "1",
            "SNAPSHOTS_ALREADY_PREPARED": "1",
            "ONLY_DECISION_COUNT": os.getenv("ONLY_DECISION_COUNT", "1"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "main.py"), str(CONFIG)]
    print(f"Launch: {' '.join(cmd)}")
    print(f"Monitor: {DURATION_SEC}s every {SAMPLE_INTERVAL}s | ONLY_DECISION_COUNT={env['ONLY_DECISION_COUNT']}")
    child = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    samples: list[dict] = []
    deadline = time.time() + DURATION_SEC
    try:
        while time.time() < deadline:
            rows = _project_pythons()
            total_cpu = sum(r[1] for r in rows)
            total_mem = sum(r[2] for r in rows)
            sample = {
                "t": round(time.time(), 1),
                "workers": len(rows),
                "sum_cpu_pct": round(total_cpu, 1),
                "sum_mem_pct": round(total_mem, 1),
                "pids": [r[0] for r in rows],
            }
            samples.append(sample)
            print(
                f"[{len(samples):03d}] workers={sample['workers']:2d} "
                f"cpu_sum={sample['sum_cpu_pct']:6.1f}% mem_sum={sample['sum_mem_pct']:5.1f}%",
                flush=True,
            )
            time.sleep(SAMPLE_INTERVAL)
    finally:
        rows = _project_pythons()
        for pid, *_ in rows:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()

    if not samples:
        print("No samples collected")
        return 1

    peak = max(samples, key=lambda s: s["sum_cpu_pct"])
    avg_cpu = sum(s["sum_cpu_pct"] for s in samples) / len(samples)
    peak_workers = max(s["workers"] for s in samples)
    report = {
        "machine_logical_cpus": int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"]).decode().strip()),
        "duration_sec": DURATION_SEC,
        "samples": len(samples),
        "peak_workers": peak_workers,
        "peak_sum_cpu_pct": peak["sum_cpu_pct"],
        "peak_workers_at_peak": peak["workers"],
        "avg_sum_cpu_pct": round(avg_cpu, 1),
        "peak_mem_sum_pct": max(s["sum_mem_pct"] for s in samples),
        "note": "macOS ps pcpu is per-core; sum ~800% means ~8 cores fully used",
    }
    out = PROJECT_ROOT / "jobs" / "cpu_probe_10models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "samples": samples[-20:]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
