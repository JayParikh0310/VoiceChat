# Embedding wrapper — loads all-MiniLM-L6-v2 once and exposes embed(text) -> vector.
# Singleton pattern: model loaded once at startup, reused across all memory ops.
# Implementation: Part 3
