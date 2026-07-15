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
| TTS per-sentence synthesis (Kokoro-82M, CUDA) | **~120 ms** | — | Warm, 3 test sentences (13-16 words each): 167ms/109ms/98ms. Measured via `tests/test_tts.py::test_synthesize_latency_within_budget` (5 runs, asserts <300ms). Requires CUDA torch — see note below; on CPU-only torch this was 1200-1500ms/sentence, ~10x slower. |
| **End-to-end total** | **pending Part 7** | — | **mic → first audio out** |

## Analysis

STT, LLM/memory, and TTS are all benchmarked now (Parts 1-4). 68ms STT + 25ms retrieve + 587ms LLM first token + ~120ms first-sentence TTS ≈ **~800ms** to first audio actually leaving the speaker, warm — well inside the 2s budget, before VAD, guardrails, or pipeline overhead (Part 7) are added.

**Environment note — Windows CUDA runtime gap (recurring theme)**: this is the *second* time in this project a default `pip`/`uv` install silently gave a CPU-only build on Windows where a GPU one was expected — same root cause class as the STT cuBLAS/cuDNN issue in Part 1.
- STT (Part 1): `faster-whisper`/CTranslate2 needed `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` — fixed by prepending their DLL dirs to `PATH` in `audio/stt.py`.
- TTS (Part 4): `torch` itself installs CPU-only from the default PyPI index on Windows (unlike Linux, where the default wheel bundles CUDA) — this silently made Kokoro synthesis ~10x slower without erroring. Fixed in `pyproject.toml` via `[tool.uv.sources]`/`[[tool.uv.index]]`, scoping just `torch`/`torchaudio` to PyTorch's `cu126` index (exact version pin `==2.6.0` on both — a mismatched `torchaudio` version against a pinned `torch` build crashes on import with `OSError: [WinError 127]`, a second bug found while fixing the first).
- **Takeaway for later parts**: if anything CPU-bound and unexpectedly slow shows up again, check `torch.cuda.is_available()` and the exact package/DLL search path before assuming it's the model or the code.

**Operational note — Ollama's `keep_alive`**: during Part 4 testing, LLM first-token jumped back up to 7.5s (from the 587ms warm baseline) after roughly 10+ idle minutes spent on large downloads. Ollama unloads a model from VRAM after its default `keep_alive` window (5 minutes) with no requests. This means "pre-warm at startup" (Part 7) isn't a one-time fix — a demo with a long pause between turns, or the CLI sitting idle before a recording starts, could still hit a cold-start. Worth considering an explicit `keep_alive` value in the Ollama call options, or a periodic background ping, when building Part 6/7.

## Does It Hit the 2-Second Budget?

On track, warm: STT (68ms) + retrieve_memory (25ms) + LLM first token (587ms) + first-sentence TTS (~120ms) ≈ 800ms of the 2000ms budget (40%), leaving ~1200ms for VAD + guardrails + pipeline overhead + audio I/O latency (Part 7). extract_memory doesn't count against this since it's fire-and-forget. The two "silent CPU fallback" bugs above are now both fixed, but the Ollama `keep_alive` finding means cold-start risk isn't fully closed out until Part 6/7 explicitly handle it.

## Implementation: Part 1 (STT), Part 2 (LLM first token, warm/cold), Part 3 (retrieve_memory latency), and Part 4 (TTS per-sentence latency) done 2026-07-14. VAD numeric benchmark and full end-to-end (Part 7) pipeline still pending.
