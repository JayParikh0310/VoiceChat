"""
Per-component latency benchmark. Grows with each Part.

Part 1: STT benchmark
Part 4: TTS benchmark (added later)
Part 6: Full pipeline benchmark (added later)

Usage:
    python scripts/benchmark.py --component stt
    python scripts/benchmark.py --component all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ── STT Benchmark ──────────────────────────────────────────────────────────────

def benchmark_stt(config: dict, runs: int = 20) -> None:
    from audio.stt import WhisperSTT

    model_size = config["stt"]["model_size"]
    device = config["stt"]["device"]
    compute_type = config["stt"]["compute_type"]

    print(f"\n{'─'*55}")
    print(f"  STT Benchmark")
    print(f"  Model   : faster-whisper {model_size}")
    print(f"  Device  : {device} ({compute_type})")
    print(f"  Runs    : {runs}")
    print(f"{'─'*55}")

    stt = WhisperSTT(config)

    # 5 seconds of low-amplitude white noise (Whisper handles empty audio gracefully)
    sample_rate = config["audio"]["sample_rate"]
    test_audio = (np.random.randn(5 * sample_rate) * 0.05).astype(np.float32)

    # Warm-up run (first inference is always slower due to CUDA/kernel init)
    print("  Warming up...")
    stt.transcribe(test_audio)

    latencies: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        stt.transcribe(test_audio)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  Run {i+1:02d}: {ms:.0f}ms")

    latencies.sort()
    median = latencies[len(latencies) // 2]
    p95    = latencies[int(len(latencies) * 0.95)]
    p_min  = latencies[0]
    p_max  = latencies[-1]

    # Budget: STT should consume < 400ms of the 2s total budget
    budget_ms = 400
    status = "✓ PASS" if median < budget_ms else "✗ FAIL — consider smaller model or CPU→CUDA"

    print(f"\n  {'─'*45}")
    print(f"  Median  : {median:.0f}ms")
    print(f"  P95     : {p95:.0f}ms")
    print(f"  Min/Max : {p_min:.0f}ms / {p_max:.0f}ms")
    print(f"  Budget  : < {budget_ms}ms")
    print(f"  Status  : {status}")
    print(f"  {'─'*45}\n")

    # Remind user to paste results into latency_report.md
    print("  → Paste these numbers into docs/latency_report.md (Part 9)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceChat component latency benchmarks.")
    parser.add_argument(
        "--component",
        choices=["stt", "tts", "all"],
        default="stt",
        help="Which component to benchmark (tts added in Part 4)",
    )
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    config = load_config()

    if args.component in ("stt", "all"):
        benchmark_stt(config, runs=args.runs)

    if args.component in ("tts", "all"):
        print("\n[TTS] TTS benchmark added in Part 4.")

    if args.component == "all":
        print("\nFull pipeline benchmark added in Part 6.")


if __name__ == "__main__":
    main()
