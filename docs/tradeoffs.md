# Design Decisions & Tradeoffs

## 1. Why Ollama over raw llama.cpp or HuggingFace transformers?
Decision: Ollama
Tradeoff: Slightly more overhead than raw llama.cpp, but saves ~4 hours of setup time on a 3-day deadline. The latency difference (< 50ms) doesn't affect the 2s budget meaningfully.

## 2. Why sentence-level TTS streaming over word-level?
Decision: Sentence boundaries
Tradeoff: Word-level would feel jittery/robotic due to short audio segments and cross-word prosody breaks. Sentence-level gives natural speech rhythm while still starting audio output well before LLM finishes.

## 3. Why input + output guardrails only (no dialogue rails)?
Decision: Skip dialogue rails
Tradeoff: Dialogue rails add 200-400ms per turn — impossible to hit 2s budget with them enabled. Input + output rails catch the safety-critical cases at acceptable cost (~50ms each).

## 4. Why async memory extraction?
Decision: Fire-and-forget in parallel
Tradeoff: Means facts from the current turn aren't retrievable until the *next* turn. Acceptable — retrieval is most useful for facts from past turns, not the current one.

## 5. Why Kokoro over Piper?
Decision: Kokoro-82M (primary)
Tradeoff: Kokoro has noticeably more natural voice quality. Piper is ~30ms faster for first chunk. Voice quality matters more for a voice assistant demo — kept Piper as configurable fallback.

## 6. Why 3B parameter model over 7B?
Decision: 3B (Qwen2.5-3B or phi-3.5-mini)
Tradeoff: 7B first-token latency on CPU easily exceeds 1s, blowing the 2s budget. 3B models have sufficient capability for conversational use; the 2s constraint is the binding constraint, not output quality.

## 7. Why FAISS over Chroma?
Decision: FAISS
Tradeoff: FAISS has no built-in metadata storage (hence SQLite alongside it), but it's faster and has zero server overhead. Chroma would be simpler to query with metadata filters but adds complexity for a small fact store.

## Implementation: Part 10
