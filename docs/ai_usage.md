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

## Documentation
- [Fill in]

## Demo Video
- [Fill in]

---
*All AI assistance is disclosed above. Core engineering decisions, implementation, debugging, and integration were done by Jay Parikh.*

## Implementation: Part 10 (fill in as you build)
