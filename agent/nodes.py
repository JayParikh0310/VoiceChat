# LangGraph node implementations:
#   generate_node       — streaming LLM call via Ollama, emits sentence chunks
#   retrieve_memory_node — embed query, search FAISS, inject top-k facts into context
#   extract_memory_node  — classify if turn contains storable fact, write to store (async)
# Implementation: Part 2 (generate_node) → Part 3 (memory nodes)
