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
- Nodes: retrieve_memory → generate → (async) extract_memory
- Role: Core reasoning layer with memory access

### Memory System
- Short-term: rolling deque of last 10 turns (in-memory)
- Long-term: FAISS vector store + SQLite (persisted to data/memory/)
- Embeddings: all-MiniLM-L6-v2

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
