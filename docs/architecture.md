# Architecture

## Pipeline Flow

[Diagram goes here — see architecture.excalidraw]

## Component Breakdown

### VAD (Voice Activity Detection)
- Tech: Silero VAD
- Role: Detects speech start/end in real-time mic stream
- Output: audio chunk when user finishes speaking

### STT (Speech-to-Text)
- Tech: faster-whisper (small, int8, CUDA)
- Role: Transcribes audio chunk to text
- Latency target: < 400ms — measured 68ms median / 81ms P95 on RTX 4060 Laptop (see `latency_report.md`)

### LangGraph Agent
- Tech: LangGraph + Ollama (qwen2.5:3b)
- Nodes: `START → retrieve_memory → generate → extract_memory → END` — fully wired as of Part 3.
- Role: Core reasoning layer with memory access
- Streaming: `generate_node` uses LangGraph's `get_stream_writer()` to emit raw tokens via `stream_mode="custom"` as they arrive from Ollama, rather than collecting the full response first — this is what lets the sentence chunker (Part 4) start TTS mid-generation.
- Measured: 587ms warm first-token latency, ~36s cold (see `latency_report.md`) — confirms Ollama pre-warm is required at pipeline startup (Part 7).
- `extract_memory` is sequenced last in the graph but is fire-and-forget internally (`asyncio.create_task`) — the node returns near-instantly and the actual classify+store work runs in the background, so it doesn't add to turn latency. Facts from the current turn aren't retrievable until the next turn.

### Memory System
- Short-term: `memory/short_term.py`'s `ShortTermMemory` — rolling deque of last 10 turns (in-memory), owned by the pipeline (Part 7), not the graph. `AgentState["messages"]` is a snapshot passed in/out each turn.
- Long-term: `memory/long_term.py`'s `LongTermMemory` — FAISS (`IndexIDMap` over `IndexFlatIP`, cosine similarity) + SQLite (fact text), persisted to `data/memory/`. Same integer id used in both, kept in sync on every write. Includes near-duplicate detection (cosine ≥ 0.92 skips re-storing).
- Embeddings: `memory/embeddings.py`'s `Embedder` — all-MiniLM-L6-v2, normalized vectors, loaded once.
- Fact extraction: a second, separate Ollama call (not the conversational one) classifies whether a user utterance contains a durable fact and rewrites it in third person, or returns `NONE`. Deliberately classifies on the user's text only, not the assistant's reply — see `agent/prompts.py` for why. See `tradeoffs.md` for the full retrieval/extraction design writeup and known limitations.

### Guardrails
- Tech: NeMo Guardrails, `guardrails/manager.py`'s `GuardrailsManager` — `check_input(text)` / `check_output(text)`, both `async`, each returning a `GuardrailResult(allowed, message)`.
- Scope: input + output self-check rails only (no dialogue rails — latency budget). Config: `guardrails/config.yml` (main model = qwen2.5:3b via Ollama, `self_check_input`/`self_check_output` few-shot prompts) + `guardrails/rails/input_rails.co` / `output_rails.co` (Colang flow definitions, kept as the canonical human-readable spec — see below for why they aren't what actually runs).
- **How a check runs**: `GuardrailsManager` calls NeMo's built-in `self_check_input`/`self_check_output` actions directly via `LLMRails.runtime.action_dispatcher.execute_action(...)`, bypassing `rails.generate()` and the Colang flow engine entirely. This was a deliberate choice, not a simplification for its own sake: `rails.generate()` drives NeMo's own dialog loop, and with nothing to stop it, an allowed input falls through to NeMo generating its *own* full response with the main model — redundant, since `agent/nodes.py`'s `generate_node` already owns real response generation with memory-aware context. Direct action dispatch gets exactly one focused LLM call per check and nothing else. See `docs/tradeoffs.md` for the debugging path that led here (an `options={"rails": {...}}`-restricted `generate()` call was tried first and silently no-op'd on output checks).
- **Fail-open on error**: if the self-check action itself fails (Ollama down, exception, etc.), `check_input`/`check_output` return `allowed=True` rather than blocking — a broken safety check shouldn't make the assistant go silent. Consistent with this project's "no abrupt silences" UX requirement; the tradeoff is a check that's down fails permissively rather than restrictively. Config toggles (`guardrails.enabled`, `check_input`, `check_output` in `config.yaml`) can disable checks entirely, in which case no LLM call is made at all and the manager just returns `allowed=True` immediately.
- **Not yet wired into the pipeline** — that's explicitly Part 7's job per the plan ("guardrails wired at pipeline level, not inside LangGraph graph, to keep the graph clean and avoid adding latency inside the LLM generation path"). Part 5 built and tested the checks in isolation only. See `docs/latency_report.md` for the real measured cost of wiring them in (it's non-trivial — ~328ms input, ~324ms output per sentence).

### TTS (Text-to-Speech)
- Tech: Kokoro-82M via `audio/tts.py`'s `KokoroTTS` — 24kHz output, independent of the 16kHz mic/STT rate (`AudioManager.play_audio()`/`open_speaker()` open the output stream at whatever rate the TTS engine reports, reopening only if the rate changes). Piper (fallback) is config-selectable (`create_tts_engine()`) but not implemented — `piper-tts` isn't installed and wasn't needed once Kokoro's CUDA issue was fixed; raises `NotImplementedError` if selected.
- `SentenceChunker`: regex-based (`[.!?]\s`) boundary detection on the raw token stream. Requires punctuation *followed by whitespace*, which avoids false-splitting decimals ("3.14") but does split mid-sentence abbreviations ("Dr. Smith") — an accepted simplification, not full NLP sentence segmentation. See `tradeoffs.md`.
- `SentenceStreamPlayer`: the overlap orchestrator — a producer coroutine feeds tokens into the chunker and queues completed sentences; a consumer coroutine pulls off that queue and does synthesis + playback (each wrapped in `asyncio.to_thread` since both Kokoro inference and PyAudio writes are blocking calls). The two run concurrently via `asyncio.gather`, so sentence 2 can be synthesizing while sentence 1 is still playing.
- Measured (CUDA torch): ~100-170ms per average sentence. On CPU-only torch this was 1200-1500ms/sentence — see `latency_report.md`'s CUDA runtime gap note for why, and the fix.

### Latency Monitor
- `pipeline/latency_monitor.py`'s `LatencyMonitor` — per-turn stage timer. `start_turn()` resets, `mark(stage)` records elapsed ms since `start_turn()` at each checkpoint (called in the order stages actually happen — `stt`, `llm_first_token`, `tts_first_chunk`, etc.), `log_summary()` derives each stage's individual duration from the gap between consecutive marks and logs a breakdown like `[stt: 68ms | llm_first_token: 587ms | total: 655ms]`, gated on `config.yaml`'s `logging.log_latency`.
- Deliberately mark()-based rather than a context-manager-per-stage: every stage the plan lists (VAD end, STT end, LLM first token, TTS first chunk, TTS complete) is a point-in-time event in an inherently sequential pipeline, where one stage's end is the next stage's start — a single ordered dict of checkpoints is enough to reconstruct every stage's duration, so a second API wasn't needed.
- Not yet wired into a live turn (that's Part 7); Part 6 verified it standalone and via `pipeline/fallback.py`'s race (see `docs/tradeoffs.md`). Reused for both the fallback threshold check and the Part 9 benchmark report, per the plan.

### Fallback Manager
- `pipeline/fallback.py`'s `FallbackManager` — `watch(first_token_event: asyncio.Event) -> bool`, run concurrently with the real streaming generation call (`asyncio.gather(fallback.watch(event), generate(...))`, with `generate` setting the event on its first token). Races `asyncio.wait_for(event.wait(), timeout=...)` against `fallback.llm_first_token_timeout_ms` (800ms default). If the event wins, `watch()` returns `False` and does nothing. If the timeout wins, `watch()` picks a random phrase from `fallback.filler_phrases`, synthesizes it with the real `KokoroTTS` engine, plays it through the real `AudioManager`, and returns `True`.
- Fires filler phrases if LLM first-token latency > 800ms. Proactive, not reactive — fires on delay, not on error; the assistant should sound like it's still "with you" during a slow response rather than going silent or announcing a failure.
- Directly depends on the real `audio.tts.KokoroTTS` + `audio.audio_manager.AudioManager` (not a generic callback abstraction) since both were already built and tested in Part 4 — consistent with how `SentenceStreamPlayer` uses them directly.
- Not yet wired into the pipeline's real generation call — that's Part 7. Part 6 built and tested the race + phrase-speaking mechanism in isolation, verified with a real concurrent `asyncio.gather()` against a simulated slow/fast task and confirmed audible playback.

## Design Decisions & Tradeoffs

[See tradeoffs.md]

## Implementation: Part 10
