# AI Usage Disclosure

This document specifies where AI tools were used in this project, per submission requirements.

## Architecture & Planning
- **Tool**: Claude (claude.ai + Claude Code)
- **Where**: Initial architecture design, tech stack selection, file structure planning (see chat.md for the full conversation), Sprint 0 CLAUDE.md creation
- **What was used**: Architectural recommendations, component selection rationale, file organization

## Part 1 — Audio Foundation (VAD + STT)
- **Tool**: Claude Code
- **Where**: Environment setup (`uv` project bootstrap, dependency install, `ollama pull qwen2.5:3b`), running and diagnosing `tests/test_vad.py` / `tests/test_stt.py` / `scripts/benchmark.py`
- **What was used**: Diagnosed a Windows-specific CTranslate2 CUDA runtime bug (missing cuBLAS/cuDNN DLLs) and implemented the fix in `audio/stt.py` (Windows-only DLL path registration, ~10 lines) — approved explicitly before writing, per this file's code-change rule. `audio/vad.py`, `audio/stt.py` (core logic), and `audio/audio_manager.py` were implemented in an earlier session (not this one).

## Code
- Part 1: see above — one targeted fix (`audio/stt.py`) for a CUDA DLL loading bug found during testing, explicitly requested before implementation.
- Part 2 — LLM Core (2026-07-14): implemented `agent/state.py` (`AgentState` TypedDict), `agent/prompts.py` (system prompt + `build_system_prompt()` formatter), `agent/nodes.py` (`make_generate_node()` — streaming Ollama call via LangGraph's custom stream writer), `agent/graph.py` (`build_graph()` — single-node `START → generate → END` graph). Explicitly requested by Jay, one part at a time. Verified via a standalone script (hardcoded prompt → graph → tokens streamed to console); not committed as a repo file since it wasn't part of the Part 2 file list.
- Part 3 — Memory System (2026-07-14): implemented `memory/embeddings.py` (`Embedder`), `memory/short_term.py` (`ShortTermMemory`), `memory/long_term.py` (`LongTermMemory` — FAISS + SQLite, with near-duplicate detection), `agent/nodes.py` additions (`make_retrieve_memory_node`, `make_extract_memory_node`), `agent/graph.py` update (full 3-node graph), `agent/prompts.py` additions (`EXTRACTION_SYSTEM_PROMPT`, `build_extraction_prompt()`), and `tests/test_memory.py` (14 tests). Explicitly requested by Jay. Involved substantial empirical prompt iteration for the memory-extraction classifier against qwen2.5:3b (documented in `tradeoffs.md`, decisions 8-9) and a real bug fix (SQLite cross-thread access via `asyncio.to_thread`) plus a similarity-threshold recalibration (`tradeoffs.md` decision 10) both found by a two-turn end-to-end integration script, not the unit tests alone. Verified via `pytest tests/test_memory.py` (14/14) and a standalone two-turn integration script (fact stated in turn 1 → recalled and used correctly in turn 2).
- Part 4 — TTS (2026-07-14): implemented `audio/tts.py` (`SentenceChunker`, `KokoroTTS`, `create_tts_engine()`, `SentenceStreamPlayer`), `audio/audio_manager.py` additions (`open_speaker()`, `play_audio()`, `close_speaker()`), and `tests/test_tts.py` (16 tests). Explicitly requested by Jay, after first testing Kokoro standalone per his request. Found and fixed a real perf bug (Windows `torch` installs CPU-only by default, making Kokoro ~10x slower than expected — `tradeoffs.md` decision 11) by scoping `torch`/`torchaudio` to PyTorch's CUDA index in `pyproject.toml`, plus a second bug that surfaced while fixing the first (mismatched `torch`/`torchaudio` versions crashing on import). Verified via `pytest tests/test_tts.py` (16/16, includes a fake-token-stream test of the overlap orchestration) and a standalone end-to-end script: real LLM tokens from the Part 3 graph → sentence chunker → real Kokoro synthesis → real speaker playback (audible, confirmed by Jay listening).

## Documentation
- [Fill in]

## Demo Video
- [Fill in]

---
*All AI assistance is disclosed above. Core engineering decisions, implementation, debugging, and integration were done by Jay Parikh.*

## Implementation: Part 10 (fill in as you build)
