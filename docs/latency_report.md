# Latency Benchmark Report

## Hardware
- CPU: [Fill in]
- GPU: [Fill in — or "CPU only"]
- RAM: [Fill in]

## Per-Component Benchmarks (median over 20 runs)

| Component | Median Latency | P95 Latency | Notes |
|---|---|---|---|
| VAD | - ms | - ms | silence detection only |
| STT (faster-whisper base int8) | - ms | - ms | 5s test audio clip |
| LLM first token (qwen2.5:3b via Ollama) | - ms | - ms | 20-token prompt |
| TTS first chunk (Kokoro-82M) | - ms | - ms | 1-sentence input |
| **End-to-end total** | **- ms** | **- ms** | **mic → first audio out** |

## Analysis

[Fill in after running scripts/benchmark.py + test_latency.py]

## Does It Hit the 2-Second Budget?

[Fill in]

## Implementation: Part 9 (fill in after benchmarks run)
