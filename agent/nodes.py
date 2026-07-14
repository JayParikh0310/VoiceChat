# LangGraph node implementations:
#   generate_node       — streaming LLM call via Ollama, emits sentence chunks
#   retrieve_memory_node — embed query, search FAISS, inject top-k facts into context
#   extract_memory_node  — classify if turn contains storable fact, write to store (async)
# Implementation: Part 2 (generate_node) → Part 3 (memory nodes)
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import ollama
from langgraph.config import get_stream_writer

from agent.prompts import build_extraction_prompt, build_system_prompt, EXTRACTION_SYSTEM_PROMPT
from agent.state import AgentState
from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

# Keeps references to extract_memory_node's fire-and-forget background tasks so they
# aren't garbage-collected mid-run (asyncio only holds a weak reference otherwise).
_background_tasks: set[asyncio.Task] = set()


def make_generate_node(config: dict) -> Callable[[AgentState], Awaitable[dict]]:
    """Build generate_node, closing over one reused AsyncClient + the configured
    model/options so they're created once at graph-build time, not per turn.
    """
    llm_cfg = config["llm"]
    client = ollama.AsyncClient(host=llm_cfg["base_url"])
    model: str = llm_cfg["model"]
    options = {
        "temperature": llm_cfg["temperature"],
        "num_predict": llm_cfg["max_tokens"],
    }

    async def generate_node(state: AgentState) -> dict:
        """Stream a response from Ollama for state["transcript"].

        Emits each raw token via LangGraph's custom stream writer as it arrives
        (consumed with graph.astream(..., stream_mode="custom")) so the sentence
        chunker (Part 4) can start TTS before generation finishes, instead of
        waiting for the full response.
        """
        writer = get_stream_writer()

        history = state.get("messages", [])
        user_turn = {"role": "user", "content": state["transcript"]}
        request_messages = [
            {"role": "system", "content": build_system_prompt(state.get("retrieved_facts"))},
            *history,
            user_turn,
        ]

        t0 = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []

        stream = await client.chat(
            model=model,
            messages=request_messages,
            stream=True,
            options=options,
        )
        async for part in stream:
            token = part["message"]["content"]
            if not token:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
                logger.info("LLM first token [%.0fms]", (first_token_at - t0) * 1000)
            chunks.append(token)
            writer(token)

        full_response = "".join(chunks)
        logger.info(
            "LLM generation complete [%.0fms total, %d chars]",
            (time.perf_counter() - t0) * 1000,
            len(full_response),
        )

        assistant_turn = {"role": "assistant", "content": full_response}
        return {
            "messages": history + [user_turn, assistant_turn],
            "response_chunks": chunks,
        }

    return generate_node


def make_retrieve_memory_node(long_term: LongTermMemory) -> Callable[[AgentState], Awaitable[dict]]:
    """Build retrieve_memory_node, closing over the shared LongTermMemory instance."""

    async def retrieve_memory_node(state: AgentState) -> dict:
        """Embed state["transcript"], search long-term memory, inject top-k facts."""
        facts = await asyncio.to_thread(long_term.retrieve_similar, state["transcript"])
        if facts:
            logger.info("Retrieved %d fact(s) from long-term memory", len(facts))
        return {"retrieved_facts": facts}

    return retrieve_memory_node


def make_extract_memory_node(
    config: dict, long_term: LongTermMemory
) -> Callable[[AgentState], Awaitable[dict]]:
    """Build extract_memory_node, closing over a dedicated Ollama client + LongTermMemory.

    Fires the actual classify-and-store work as a background asyncio task and returns
    immediately — the graph never waits on it. Facts from this turn are therefore not
    retrievable until the *next* turn; acceptable since retrieval is most useful for
    facts from past turns/sessions, not the one just generated.
    """
    llm_cfg = config["llm"]
    client = ollama.AsyncClient(host=llm_cfg["base_url"])
    model: str = llm_cfg["model"]

    async def _classify_and_store(user_text: str) -> None:
        try:
            response = await client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": build_extraction_prompt(user_text)},
                ],
                stream=False,
                options={"temperature": 0.0, "num_predict": 60},
            )
            fact = response["message"]["content"].strip()
            if fact and fact.upper().rstrip(".") != "NONE":
                await asyncio.to_thread(long_term.store_fact, fact)
                logger.info("Memory extracted: '%s'", fact)
            else:
                logger.debug("Memory extraction: nothing worth storing this turn")
        except Exception:
            logger.exception("extract_memory_node background task failed")

    async def extract_memory_node(state: AgentState) -> dict:
        messages = state["messages"]
        if len(messages) < 2:
            return {}
        user_text = messages[-2]["content"]

        task = asyncio.create_task(_classify_and_store(user_text))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        return {}

    return extract_memory_node
