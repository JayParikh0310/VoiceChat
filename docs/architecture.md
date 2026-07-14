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
- Tech: Kokoro-82M (primary), Piper (fallback)
- Sentence-level streaming: first sentence plays while LLM generates the second

### Fallback Manager
- Fires filler phrases if LLM first-token latency > 800ms
- Proactive, not reactive — fires on delay, not on error

## Design Decisions & Tradeoffs

[See tradeoffs.md]

## Implementation: Part 10
