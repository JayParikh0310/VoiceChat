
AI Engineering Intern
Objective
Build a real-time Audio-In, Audio-Out Conversational AI Assistant that accepts user queries
through voice input and responds with natural voice output.
The assistant should deliver responses with minimal latency, providing a smooth and engaging
conversational experience. An offline implementation is preferred; however, an online
solution is acceptable if an offline approach is not feasible. In situations where the primary
response cannot be generated immediately (e.g., processing delays or service interruptions),
the assistant should intelligently handle the interaction by keeping the user engaged rather than
leaving them waiting.
Requirements
● Accept user queries through voice input.
● Respond with voice output.
● Ensure the end-to-end response time is within 2 seconds, wherever possible.
● Implement a fallback conversation flow to handle delays or interruptions gracefully.
● An offline implementation is preferred. If not feasible, an online implementation may be
used.
● The fallback should maintain user engagement without displaying generic error
messages or abruptly ending the conversation.
● Design the solution with a focus on responsiveness, user experience, and natural
conversation.
Submission
● Upload all deliverables to Google drive.
● Mention the Use of the AI properly, specifying where it was used.
● Create a document containing the Google Drive links to all deliverables.
● Submit this document in PDF format.
● Ensure that all links are accessible and working. We will not contact you if any of the
submitted links are inaccessible.

I want to do the ai one obviously, ignore the rest read it and rate the difficulty, I have about 3-4 days to do it, I want to make it offline, how will it work offline, I get it I can download llama model in laptop and not use API's but the latency for normal chat with that only will excede more than 2 seconds even with stream, to convert voice to language I can use wisperflow but is that free and usable offline, also how to convert llm responce to voice offline nicely, and to make it more secure and stuff and better I want to make guardrails and long term memory also, don't make anything just address my points and tell me

Jay Parikh Resume.pdf
pdf

Here go ahead


Evaluated portfolio gaps and strategized differentiation approach
Evaluated portfolio gaps and strategized differentiation approach
Looked at it — here's the honest read.

What's already there: You've got three GenAI/agentic projects (Incident Simulator, Agentic & Generative AI Suite, AI Learning Platform) plus the NxtGen internship, all built on LangChain/LangGraph, RAG, and LLM-app patterns. Your resume already signals "I can build agentic LLM apps" strongly — arguably that box is checked multiple times over.

Where this project actually helps: It's not "another chatbot" if you frame it right, because it fills gaps your current resume doesn't touch at all:

Real-time/latency-constrained systems — nothing on your resume currently shows you optimizing for hard latency budgets. This does.
Speech/audio domain — zero audio-related work currently. STT/TTS pipeline is a genuinely different skill surface than text-based RAG.
Offline/on-device deployment — everything else you've built is API/cloud-service oriented (FastAPI, Render/Vercel/Neon). Local model deployment (quantization, llama.cpp/Ollama, on-device inference tradeoffs) is a different, valued skill, especially for a robotics-adjacent company where on-device constraints actually matter.
Where it could backfire: If you write the bullet points generically ("built a voice assistant using Whisper and LLM"), it will blend into the sea of similar submissions and look repetitive next to your existing agentic projects. The differentiation has to come through in how you describe it — lead with the engineering decisions, not the feature list:

"Engineered sub-2s end-to-end voice pipeline via sentence-level response streaming, overlapping LLM generation with TTS synthesis"
"Deployed fully offline (faster-whisper + quantized local LLM + Piper/Kokoro TTS), no cloud dependency"
"Implemented dual-memory architecture — async fact-extraction into a long-term vector store + short-term context buffer — with similarity-gated retrieval for personalization"
"Scoped NeMo Guardrails to input/output rails only to preserve latency budget, documented the tradeoff"
That kind of phrasing signals systems thinking, not "did the assignment." Given your existing project spread already leans heavily text/agentic, this genuinely rounds you out rather than duplicating what's there — worth doing, just don't undersell the engineering reasoning when you write it up.

In the Google form to submit its asking to just submit on file of 1mb max pdf format for the project, they are not even checking the actual project wtf, just a pdf I can't even put many screenshots after I make the project


Clarified PDF submission contains links, not full project files
Clarified PDF submission contains links, not full project files
That's not them being lazy — re-read the submission requirement: the PDF is just supposed to be a links document, not the project showcase itself. The brief literally says "Create a document containing the Google Drive links to all deliverables." The actual evaluation happens through what's behind those links — your demo video, GitHub repo, code, docs — all uploaded to Drive separately. The PDF is just an index/pointer file, so 1MB is intentionally tiny because it's not meant to hold your actual work.

So don't try to cram screenshots into the PDF. Structure it like this:

1-2 lines per deliverable type (repo, demo video, architecture doc, etc.)
Direct Drive links for each (make sure sharing is set to "anyone with link can view")
Maybe one small compressed screenshot or a simple architecture diagram if you want visual context — that alone will fit in 1MB easily since it's just text + a couple links + one lightweight image
Where your actual project gets shown off is:

A demo video (even 2-3 min, screen-recorded) walking through voice-in → response → voice-out, showing the fallback flow and ideally a quick look at memory retrieval working — this is what actually gets watched.
Your GitHub repo with a clean README (architecture diagram, tradeoffs, setup instructions) — this is what a technical reviewer will actually dig into if you make round 1.
Drive folder can also hold extra screenshots, logs, latency benchmarks, whatever — no size constraint there.
So nothing's actually being wasted — you just put the "wow factor" in the video + repo, and keep the PDF as a clean, professional index. If anything, a lot of applicants will mess this up by dumping everything into the PDF and hitting the 1MB wall — you won't.

Ohh ok got it, let's start with the project first, let's build the architecture first and foremost, I am sure it won't be as complex as incident simulator or smtn, give me decent architecture idea that works for the current scope and maybe if I want to expand in any way in the future


Architected scalable voice assistant framework with extensible components
Architected scalable voice assistant framework with extensible components
Good instinct — this doesn't need Incident Simulator-level complexity. That project was a full-stack multi-service app; this is a pipeline with feedback loops, more like a modified LangGraph agent with audio I/O bolted onto both ends. Here's the shape of it:

Core flow (per turn):

Mic input → VAD (detect speech start/end) → faster-whisper (STT)
    → [Guardrail: input rail check] 
    → LangGraph agent:
         ├─ retrieve_memory node (embed query → search long-term fact store)
         ├─ generate node (streaming LLM call, context = short-term buffer + retrieved facts)
         └─ extract_memory node (async, runs in parallel, doesn't block response)
    → sentence-chunker (splits streamed tokens into complete sentences)
    → [Guardrail: output rail check, per sentence-chunk]
    → Piper/Kokoro TTS (streams audio per chunk as it's cleared)
    → Speaker output
Components, mapped to your existing skill set:

Layer	Tech	Why
STT	faster-whisper (small/base, int8)	You already know Python; CTranslate2 backend is fast on CPU/GPU
VAD	Silero VAD or webrtcvad	Cheap, tells you when user stopped talking — needed to trigger STT
Orchestration	LangGraph	You already use this daily — map nodes exactly as above
LLM	Qwen2.5-3B-Instruct or Llama-3.2-3B via Ollama/llama.cpp	Small enough for real-time on modest hardware
Short-term memory	Simple in-memory deque / LangGraph state	Rolling last N turns
Long-term memory	FAISS/Chroma + all-MiniLM embeddings, SQLite for raw facts	Matches your "facts about user" spec from earlier
Guardrails	NeMo Guardrails (input/output rails only)	As discussed — skip dialogue rails for latency
TTS	Piper (fast path) or Kokoro-82M (nicer voice)	CPU-runnable, streaming-friendly
Glue/UI	FastAPI backend + simple web frontend (mic button, waveform) or just a terminal/CLI demo if time's tight	You already know FastAPI from Incident Simulator
Two memory nodes are the interesting engineering bit — worth diagramming clearly in your submission:

extract_memory: fires async after each user turn, cheap classification prompt ("is there a fact/preference here worth storing?"), writes to vector store if yes.
retrieve_memory: fires before generation, embeds the current query, does similarity search against the fact store, injects top-k above a threshold into context.
Future expansion points worth mentioning in your doc (even if unbuilt), shows you're thinking ahead:

Swap Piper → Kokoro for expressive TTS once latency budget allows
Multi-user support (memory store keyed by user_id)
Wake-word detection (Porcupine) instead of push-to-talk
Interrupt handling (barge-in — user speaks over TTS output, cuts it off)
Swap local 3B model for a cloud model dynamically if local hardware unavailable (graceful degrade)
Want me to help you pick a concrete demo interface next (CLI vs simple web mic UI), or lock down the exact model files/versions to install first?

Is there any other better way to make it offline rater than downloading the model locally like how we are doing now, also five file structure and all the files I will require in general, also mention if any other planning we should do before implementation


Evaluated offline approaches and architected comprehensive project planning framework
Evaluated offline approaches and architected comprehensive project planning framework
On "better way to go offline"

No — offline inherently means the model weights live on your machine; there's no way around downloading them once. That's not a limitation of your approach, it's just what "offline" means as a technical property. The only real choice is how you manage that download/load:

Ollama — pulls a quantized GGUF automatically, handles serving via simple API, least setup friction. Recommended for you given the time constraint.
llama.cpp directly — more control, more manual setup, marginal latency gain, not worth it unless Ollama's overhead becomes a real bottleneck.
Raw HF transformers + safetensors — avoid, no quantization out of the box, much slower and heavier on RAM/VRAM.
So stick with Ollama for the LLM. Same logic applies to STT (faster-whisper downloads model weights once, caches locally) and TTS (Piper/Kokoro same). "Offline" just means: after the one-time download, zero network calls happen at runtime. Worth stating that explicitly in your docs — it shows you understand the distinction between "no internet needed to build" (false) vs "no internet needed to run" (true, and what matters).

File structure

voice-assistant/
├── README.md
├── requirements.txt
├── config.yaml                  # model names, paths, thresholds, timeouts
├── main.py                      # entrypoint, ties pipeline together
├── audio/
│   ├── vad.py                   # Silero VAD wrapper
│   ├── stt.py                   # faster-whisper wrapper
│   └── tts.py                   # Piper/Kokoro wrapper + sentence-chunk streaming
├── agent/
│   ├── graph.py                 # LangGraph definition (nodes + edges)
│   ├── nodes.py                 # generate_node, retrieve_memory_node, extract_memory_node
│   └── prompts.py                # system prompt, extraction prompt, fallback lines
├── memory/
│   ├── short_term.py            # rolling buffer / session state
│   ├── long_term.py             # embedding store (FAISS/Chroma) + fact CRUD
│   └── embeddings.py             # all-MiniLM wrapper
├── guardrails/
│   ├── config.yml                # NeMo Guardrails rails config
│   └── rails/                    # input/output rail definitions
├── tests/
│   ├── test_latency.py           # per-component timing benchmarks
│   └── test_pipeline_e2e.py
├── demo/
│   └── cli_demo.py               # or a simple web UI if you build one
└── docs/
    ├── architecture.png/.excalidraw
    └── writeup.md                 # design decisions, tradeoffs, AI-usage disclosure
Planning before you write any pipeline code:

Hardware check first — confirm what GPU/VRAM (if any) you actually have. This decides your model sizes (3B vs 7B, quantization level) before you pick anything else. Do this today.
Benchmark each component standalone before wiring together — measure STT latency alone, LLM first-token latency alone, TTS first-audio-chunk latency alone. Know your budget breakdown before integration, or debugging the combined pipeline later becomes guesswork.
Pick exact model files now — Qwen2.5-3B-Instruct-GGUF (via Ollama), faster-whisper base or small, Piper voice model (pick one voice file), all-MiniLM-L6-v2 for embeddings. Lock these in so you're not swapping mid-build.
Decide the demo interface early — CLI push-to-talk (fast to build, still demoable) vs a simple web mic UI (nicer for the video, more time cost). Given your timeline, CLI is a legitimate choice — the content matters more than UI polish here.
Write the fallback lines and guardrail test cases before coding — a short list of filler phrases for delay-handling, and a short list of adversarial/edge-case prompts to test your rails against. Do this on paper first so implementation has a target.
Git commit discipline — commit each component as it's working (VAD, then STT, then LLM node, then memory, then TTS, then guardrails) rather than one big commit at the end. Shows process if anyone actually opens the repo.
Start the docs/writeup.md alongside building, not after — jot down each tradeoff decision as you make it (why 3B not 7B, why input/output rails only, etc.) so you're not reconstructing your reasoning from memory on day 4.
Once you confirm your hardware, I can help you lock the exact model choices and quantization level next.






Claude is AI and can make mistakes. Please double-check responses.


