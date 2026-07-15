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
- Tech: NeMo Guardrails
- Scope: input + output rails only (no dialogue rails — latency budget)

### TTS (Text-to-Speech)
- Tech: Kokoro-82M via `audio/tts.py`'s `KokoroTTS` — 24kHz output, independent of the 16kHz mic/STT rate (`AudioManager.play_audio()`/`open_speaker()` open the output stream at whatever rate the TTS engine reports, reopening only if the rate changes). Piper (fallback) is config-selectable (`create_tts_engine()`) but not implemented — `piper-tts` isn't installed and wasn't needed once Kokoro's CUDA issue was fixed; raises `NotImplementedError` if selected.
- `SentenceChunker`: regex-based (`[.!?]\s`) boundary detection on the raw token stream. Requires punctuation *followed by whitespace*, which avoids false-splitting decimals ("3.14") but does split mid-sentence abbreviations ("Dr. Smith") — an accepted simplification, not full NLP sentence segmentation. See `tradeoffs.md`.
- `SentenceStreamPlayer`: the overlap orchestrator — a producer coroutine feeds tokens into the chunker and queues completed sentences; a consumer coroutine pulls off that queue and does synthesis + playback (each wrapped in `asyncio.to_thread` since both Kokoro inference and PyAudio writes are blocking calls). The two run concurrently via `asyncio.gather`, so sentence 2 can be synthesizing while sentence 1 is still playing.
- Measured (CUDA torch): ~100-170ms per average sentence. On CPU-only torch this was 1200-1500ms/sentence — see `latency_report.md`'s CUDA runtime gap note for why, and the fix.

### Fallback Manager
- Fires filler phrases if LLM first-token latency > 800ms
- Proactive, not reactive — fires on delay, not on error

## Design Decisions & Tradeoffs

[See tradeoffs.md]

## Implementation: Part 10
