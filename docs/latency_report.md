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
| LLM first token (qwen2.5:3b via Ollama, warm) | **587 ms** | — | Single hardcoded-prompt run via `agent/graph.py`, `stream_mode="custom"`. Cold (model not yet loaded into VRAM): **~36s** — confirms the "pre-warm at startup" decision in `tradeoffs.md` is load-bearing, not optional; Part 7 must fire a dummy prompt before the pipeline accepts real input. |
| retrieve_memory (embed query + FAISS search, on critical path) | **25 ms** | — | 10 runs, 3-fact store, `memory/long_term.py`'s `retrieve_similar()`. Negligible against the 2s budget — runs before `generate`, so it adds to (not overlaps with) the 587ms LLM number above. |
| extract_memory (classify + store, off critical path) | not on critical path | — | Fire-and-forget `asyncio.create_task` — the graph node itself returns near-instantly; the actual Ollama classification call (~0.5-1s) and store happen in the background and don't block this turn's audio output. |
| TTS first chunk (Kokoro-82M) | pending Part 4 | — | |
| **End-to-end total** | **pending Part 7** | — | **mic → first audio out** |

## Analysis

STT and the LLM/memory path are benchmarked so far (Parts 1-3). 68ms STT + 25ms retrieve + 587ms LLM first token ≈ **680ms** to first spoken token, warm — comfortably inside the 2s budget with TTS still to add. The GPU path (CUDA + int8 CTranslate2) and the memory retrieval path are both performing well within budget.

**Environment note — Windows CUDA runtime gap**: `torch` installed from the default PyPI index on Windows is CPU-only (unlike Linux, where the default wheel bundles CUDA). This didn't block STT directly (faster-whisper/CTranslate2 doesn't depend on torch's CUDA build), but CTranslate2 itself needs cuBLAS/cuDNN DLLs that aren't present without a full CUDA Toolkit install. Fixed by installing the pip-only `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` / `nvidia-cuda-nvrtc-cu12` packages and prepending their DLL directories to `PATH` at import time in `audio/stt.py` (Windows-only code path, no-op elsewhere). See `tradeoffs.md` for more detail if this needs write-up there too.

## Does It Hit the 2-Second Budget?

On track. STT (68ms) + retrieve_memory (25ms) + LLM first token (587ms warm) ≈ 680ms of the 2000ms budget (34%), leaving ~1320ms for VAD + TTS + guardrails + overhead. extract_memory doesn't count against this since it's fire-and-forget. Revisit after Part 4 (TTS first-chunk) and Part 7 (true end-to-end, mic to speaker) are benchmarked — the warm-vs-cold LLM gap also means Part 7's pre-warm step is not optional for hitting this budget on a user's first turn.

## Implementation: Part 1 (STT), Part 2 (LLM first token, warm/cold), and Part 3 (retrieve_memory latency) done 2026-07-14. VAD numeric benchmark, TTS, full pipeline still pending.
