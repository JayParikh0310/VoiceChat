# Latency Benchmark Report

## Hardware
- CPU: Intel Core i9-13900HX (13th Gen)
- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM), driver 596.08, CUDA 13.2
- RAM: 31.7 GB

## Per-Component Benchmarks (median over 20 runs)

| Component | Median Latency | P95 Latency | Notes |
|---|---|---|---|
| VAD | not yet benchmarked | — | Correctness-only in Part 1 (`tests/test_vad.py`, 8/8 passing). Silero VAD is a ~2MB model; expected negligible (<5ms/window) per `notes.md`, but no numeric benchmark exists yet — `scripts/benchmark.py` only covers STT/TTS/full-pipeline per its `--component` choices. Add a VAD section there if a hard number is needed for submission. |
| STT (faster-whisper **small**, int8, CUDA) | **68 ms** | **81 ms** | 5s synthetic test clip, `scripts/benchmark.py --component stt --runs 20`. Min/max: 58ms/81ms. Well under the 400ms budget slice. Note: config uses `small` (not `base` as originally planned in `notes.md`) since the RTX 4060's 8GB VRAM comfortably affords it — see decision note below. |
| LLM first token (qwen2.5:3b via Ollama) | pending Part 2 | — | Model pulled and ready (`ollama pull qwen2.5:3b`, confirmed via `ollama list`) |
| TTS first chunk (Kokoro-82M) | pending Part 4 | — | |
| **End-to-end total** | **pending Part 7** | — | **mic → first audio out** |

## Analysis

STT is the only component benchmarked so far (Part 1). 68ms median is far below the 400ms budget slice and below the ~80ms estimate in `notes.md` — the GPU path (CUDA + int8 CTranslate2) is performing as expected once the runtime dependency issue below was resolved. This leaves substantial headroom in the 2s end-to-end budget for the LLM and TTS stages still to come.

**Environment note — Windows CUDA runtime gap**: `torch` installed from the default PyPI index on Windows is CPU-only (unlike Linux, where the default wheel bundles CUDA). This didn't block STT directly (faster-whisper/CTranslate2 doesn't depend on torch's CUDA build), but CTranslate2 itself needs cuBLAS/cuDNN DLLs that aren't present without a full CUDA Toolkit install. Fixed by installing the pip-only `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` / `nvidia-cuda-nvrtc-cu12` packages and prepending their DLL directories to `PATH` at import time in `audio/stt.py` (Windows-only code path, no-op elsewhere). See `tradeoffs.md` for more detail if this needs write-up there too.

## Does It Hit the 2-Second Budget?

Too early to say — only STT is measured. STT alone consumes 68ms of the 2000ms budget (3.4%), leaving ~1930ms for VAD + LLM + TTS + guardrails + overhead. On track based on Part 1 numbers; revisit after Part 2 (LLM first-token) and Part 4 (TTS first-chunk) are benchmarked.

## Implementation: Part 1 (STT) done 2026-07-14. Parts 4/6/9 (VAD, TTS, full pipeline) pending.
