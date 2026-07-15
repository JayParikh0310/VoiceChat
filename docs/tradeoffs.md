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

## 8. Extraction prompt: classify on the user's utterance only, not the assistant's reply (Part 3)
Decision: `extract_memory_node`'s "is this worth remembering?" LLM call receives only the user's text.
Tradeoff: Initially the prompt included both sides of the turn (user text + assistant reply), matching the plan's "turn"-level framing. Empirically, against qwen2.5:3b, this made the classifier meaningfully *less* reliable — the exact same fact-bearing user statement flipped between correctly extracting and returning NONE depending only on how the assistant's reply happened to be phrased (e.g. ending in a follow-up question pushed it toward NONE). Dropping the assistant reply and classifying on the user's utterance alone scored 9/10 on a manual test set, vs. 5-7/9 with it included. A 3B model has limited capacity to ignore irrelevant context; less input was more reliable than more context here.

## 9. Few-shot example ordering matters more than instruction wording, for a 3B classifier (Part 3)
Decision: `EXTRACTION_SYSTEM_PROMPT` uses several few-shot examples, NONE-examples listed before positive examples.
Tradeoff: A terse zero-shot instruction ("respond NONE or one sentence") produced NONE for almost everything, including obvious facts. Adding few-shot examples fixed most of that, but example *order* had a surprisingly large effect — positive examples placed last (closer to where the model starts generating) extracted more reliably than the same examples placed first. This is a known small-model recency-bias pattern, not a bug; the current ordering was chosen empirically, not from a general principle, so if the extraction quality drifts after future prompt edits, try reordering examples before assuming the logic is broken.
Known limitation: the classifier is not perfectly reliable — reliably catches name/occupation/health/preference facts, sometimes misses relationship/life-event facts (e.g. "I just started a new job" was missed in testing). Accepted for now: this is explicitly a "cheap async call" per the original plan, not a component the 2-second latency budget depends on, since it's fire-and-forget.

## 10. Long-term memory similarity threshold: recalibrated from 0.6 to 0.15 (Part 3)
Decision: `config.yaml`'s `memory.long_term.similarity_threshold` is 0.15, not the originally-planned 0.6.
Tradeoff: Facts are deliberately stored in third person ("The user works as a machine learning engineer") so they read naturally when injected into the system prompt. But real user queries are first person ("what do I do for work?", "what's my job?"). Measured against all-MiniLM-L6-v2, that person mismatch costs a lot of cosine similarity even for a clear semantic match — genuine matches measured ~0.21-0.39, while third-person-phrased queries against the same fact measured ~0.52-0.60. At the original 0.6 threshold, realistic first-person voice queries would almost never clear the bar, silently breaking retrieval in the exact scenario the feature exists for. Unrelated first-person queries measured ~0.0-0.09 against the same fact, so 0.15 sits in the gap with margin on both sides. This was caught by the Part 3 end-to-end integration test (store a fact, then query for it in a fresh turn), not by the unit tests — the unit tests all used third-person-ish or high-similarity query phrasing and would not have caught this on their own. Worth re-checking if the embedding model is ever swapped.

## 11. torch on Windows silently installs CPU-only, even with a working GPU (Part 4)
Decision: `pyproject.toml` scopes `torch`/`torchaudio` to PyTorch's official `cu126` index via `[tool.uv.sources]`/`[[tool.uv.index]]` (`explicit = true`, so it doesn't affect resolution of any other package), pinned to the exact same version (`==2.6.0`) on both.
Tradeoff: the default PyPI index gives Windows a CPU-only `torch` wheel (Linux's default wheel bundles CUDA; Windows's doesn't) — this doesn't error, it just silently runs everything on CPU. It went unnoticed through Parts 1-3 because nothing torch-dependent was slow enough to be suspicious (Silero VAD and the embedding model are both tiny). It became obvious in Part 4: Kokoro-82M's vocoder is a real neural network, and CPU-only synthesis measured 1200-1500ms/sentence — 6-10x over the ~150-200ms estimate in `notes.md`, enough to meaningfully threaten the 2s budget once TTS is in the loop. Installing a CUDA build dropped it to ~100-170ms/sentence. Second bug found while fixing the first: pinning only `torch` to the CUDA index let `torchaudio` resolve independently to a mismatched version (2.11.0 against torch 2.6.0), which crashed on import (`OSError: [WinError 127]`) — torch/torchaudio versions must move together. Nothing in the codebase actually imports `torchaudio` directly (confirmed via grep); it's kept only because something in the dependency chain (kokoro/misaki) may still need it importable.
General lesson, now the second instance of this exact failure mode (see decision on the STT cuBLAS/cuDNN fix in `latency_report.md`): a `pip`/`uv` install succeeding is not evidence that the GPU path is actually being used. Check `torch.cuda.is_available()` explicitly rather than inferring it from "the install worked."

## 12. Sentence chunker: whitespace-gated boundary detection, not real NLP segmentation (Part 4)
Decision: `SentenceChunker` splits on `[.!?]\s` (punctuation immediately followed by whitespace) via a single regex, re-scanned on every token.
Tradeoff: this is a heuristic, not a sentence parser. It correctly avoids splitting on decimals ("3.14 " has no whitespace right after the first '.') without any special-casing, which was a nice side effect of the whitespace requirement rather than something explicitly designed for. But it will incorrectly split mid-sentence on abbreviations that are followed by a space and a capital letter ("Dr. Smith called." would split into "Dr." and "Smith called."). Accepted because: (1) the conversational system prompt (`agent/prompts.py`) already asks for short, plain spoken sentences, making abbreviations relatively rare in practice, and (2) the cost of a bad split is a slightly awkward pause in speech, not a functional break — TTS still speaks the fragment correctly, it just synthesizes "Dr." as its own short utterance. A real fix (spaCy sentence segmentation, which is already an incidental dependency via `misaki`) is possible later if this proves noticeable in the demo recording.

## Implementation: Part 10
