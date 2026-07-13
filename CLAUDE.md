# VoiceChat — CLAUDE.md

## CRITICAL RULE
**Do NOT write any implementation code unless Jay explicitly says so AND specifies which part/file to implement.** This includes: no function bodies, no class implementations, no logic. Scaffolding (empty files, imports, docstrings, type stubs) is fine when setting up structure. Sprint 0 is about planning and structure only.

---

## Project Overview

**Assignment**: AI Engineering Intern project — build a real-time Audio-In/Audio-Out Conversational AI Assistant.

**Core requirements**:
- Voice input → voice output conversation loop
- End-to-end response latency < 2 seconds wherever possible
- Fallback conversation flow (filler phrases) when latency exceeds threshold or service is interrupted
- Offline preferred (no runtime API calls after setup)
- Natural, engaging UX — no raw error messages, no abrupt silences

**Submission format**: 1-page PDF linking to Google Drive (repo, demo video, architecture doc, AI-usage disclosure). The PDF is a links index, not the project content. Actual eval happens via the Drive links.

---

## Architecture (Pipeline Flow)

```
Mic Input
    ↓
[VAD] — Silero VAD (detect speech start/end)
    ↓
[STT] — faster-whisper (base/small, int8) → transcript text
    ↓
[Input Guardrail] — NeMo Guardrails input rail
    ↓
[LangGraph Agent]
    ├── retrieve_memory_node  (embed query → FAISS similarity search → inject top-k facts)
    ├── generate_node         (streaming LLM call via Ollama, context = short-term buffer + retrieved facts)
    │       ↓ streamed tokens
    │   [Sentence Chunker]   (buffer tokens → emit complete sentences)
    │       ↓ sentence chunks
    │   [Output Guardrail]   (NeMo Guardrails per-sentence)
    │       ↓ cleared sentences
    │   [TTS]                (Piper/Kokoro → audio chunks → speaker, overlapped with generation)
    └── extract_memory_node   (async, parallel — classify if fact worth storing, write to vector store)
    ↓
Speaker Output

FALLBACK PATH: if LLM first-token latency > 800ms threshold → FallbackManager fires pre-recorded filler phrase immediately
```

**Key latency budget target** (per turn):
- VAD: ~50ms
- STT (base int8): ~200-400ms
- LLM first token (3B via Ollama): ~400-700ms
- TTS first chunk: ~100-200ms
- Total target: < 1.5s to first audio out

---

## Tech Stack (Locked)

| Layer | Tech | Rationale |
|---|---|---|
| STT | faster-whisper (base, int8) | CTranslate2 backend, fast CPU/GPU, offline |
| VAD | Silero VAD | More robust than webrtcvad in noisy environments |
| LLM | Qwen2.5-3B-Instruct or phi-3.5-mini via Ollama | Small, fast, quantized GGUF, offline after pull |
| Orchestration | LangGraph | Jay already uses this; maps cleanly to node/edge pipeline |
| Short-term memory | Python deque / LangGraph state | Rolling last-N turns, no persistence needed |
| Long-term memory | FAISS + all-MiniLM-L6-v2 + SQLite | Similarity search for user facts, embeddings fully offline |
| Guardrails | NeMo Guardrails (input + output rails only) | Dialogue rails skipped to preserve latency budget |
| TTS | Kokoro-82M (preferred) or Piper (fallback) | Kokoro: better voice quality; Piper: faster first chunk |
| Fallback | FallbackManager with filler phrase queue | Fires when LLM latency > threshold |
| Demo UI | CLI push-to-talk (primary); FastAPI + WebSocket (stretch) | CLI ships faster; web UI makes a better demo video |

**Model decision note**: Final model choice depends on Jay's hardware. GPU with VRAM → Qwen2.5-3B Q4_K_M. CPU only → phi-3.5-mini Q4_0 or smaller Qwen.

---

## Deliverables & Git Commit Checkpoints

Each part should be committed to git once working. Commit after each part — shows process in the repo, which reviewers actually value.

| Part | What's Done | Git Commit Message |
|---|---|---|
| **Part 0** | Folder structure, CLAUDE.md, config.yaml skeleton, requirements.txt | `chore: sprint 0 — project scaffold and architecture` |
| **Part 1** | VAD working standalone + STT working standalone, standalone latency benchmarks | `feat: audio layer — VAD (Silero) + STT (faster-whisper) with latency benchmarks` |
| **Part 2** | Ollama setup, LangGraph graph with generate_node, streaming LLM output to console | `feat: LLM core — LangGraph agent with streaming Ollama generation` |
| **Part 3** | Short-term memory buffer + long-term FAISS store + retrieve and extract nodes wired into graph | `feat: dual memory system — short-term buffer + long-term FAISS fact store` |
| **Part 4** | TTS integrated (Kokoro/Piper), sentence chunker, audio playback manager, overlapped generation+TTS | `feat: TTS layer — sentence-chunk streaming with overlapped LLM+TTS playback` |
| **Part 5** | NeMo Guardrails input + output rails configured and wired into pipeline | `feat: guardrails — NeMo input/output rails integrated` |
| **Part 6** | FallbackManager with filler phrases, latency monitor firing fallback on threshold breach | `feat: fallback system — filler phrase queue with latency-threshold trigger` |
| **Part 7** | Full pipeline wired end-to-end: mic → VAD → STT → guardrail → agent → TTS → speaker | `feat: full pipeline integration — end-to-end voice conversation working` |
| **Part 8** | CLI demo polished + optional web UI (FastAPI + WebSocket mic page) | `feat: demo interface — CLI push-to-talk + web mic UI` |
| **Part 9** | All tests passing, latency benchmarks documented | `test: unit tests + e2e latency benchmarks` |
| **Part 10** | Architecture doc, tradeoffs doc, AI-usage disclosure, demo video recorded | `docs: architecture, tradeoffs, AI-usage disclosure, submission prep` |

---

## Coding Guidelines

- Use Python 3.10+
- Type hints on all function signatures
- Async-first where latency matters (TTS playback, memory extraction)
- Config-driven: all model names, paths, thresholds in `config.yaml` — no hardcoded values in pipeline code
- Log timing at each pipeline stage (use Python `logging`, not print statements in production paths)
- No comments explaining what the code does — only comments for non-obvious WHY (workarounds, constraints)
- Tests: each component gets its own test file; latency tests use real models, not mocks
- Git: one commit per Part, commit message format from the table above

---

## Key Engineering Decisions (pre-made)

1. **Offline = no runtime network calls** — models downloaded once at setup, zero API calls at inference time. State this explicitly in docs.
2. **Sentence-level streaming over word-level** — word-level TTS is choppy; sentence boundaries give natural speech rhythm.
3. **Input + output guardrails only (no dialogue rails)** — dialogue rails add ~200-400ms per turn; we can't afford it given the 2s budget.
4. **extract_memory runs async** — it must not block the TTS playback path. Fire-and-forget pattern.
5. **Fallback fires at 800ms LLM threshold** — not "if error occurs" but proactively on slow response. This is the key UX insight.
6. **LLM pre-warm at startup** — send a dummy prompt on init to avoid cold-start on the first real user query.

---

## Hardware Note (fill in before Part 1)

Jay's hardware: **[TODO: fill in GPU model + VRAM, or CPU-only]**
Model size decision depends on this. Default assumption: mid-range GPU or CPU-only → Qwen2.5-3B Q4_K_M.

---

## Future Expansion Points (document in writeup, don't build)

- Barge-in / interrupt handling: user speech cuts TTS immediately
- Wake-word detection (Porcupine) instead of push-to-talk
- Multi-user support: memory store keyed by user_id
- Swap Piper → Kokoro for better voice quality once latency allows
- Dynamic fallback to cloud LLM if local hardware can't hit latency budget

---

## Submission Checklist

- [ ] GitHub repo public with clean README
- [ ] Demo video (2-3 min screen recording) on Google Drive
- [ ] Architecture diagram in `/docs/`
- [ ] Tradeoffs writeup in `/docs/`
- [ ] AI-usage disclosure in `/docs/`
- [ ] Latency benchmark report in `/docs/`
- [ ] All Drive links accessible (test before submitting)
- [ ] 1-page PDF with Drive links (max 1MB)

---

## AI Usage in This Project

*(Fill this in as you build — required for submission)*

- Claude Code (claude-sonnet-4-6): project architecture design, CLAUDE.md creation, Sprint 0 scaffolding
- Claude.ai (claude-sonnet-4-6): initial architecture brainstorm, tech stack selection, file structure planning (see `chat.md`)
- [Add per component as you use it]
