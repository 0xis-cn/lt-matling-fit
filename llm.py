# rag_agent.py  (LangChain 1.0.4 compatible)
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Iterator
from datetime import date
from pydantic import BaseModel, Field

from worker import keyword_search, semantic_search, list_archives  # DO NOT reimplement

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool, Tool
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser

import dotenv
dotenv.load_dotenv()

# ---------------- Tool input schemas ----------------
class KeywordSearchInput(BaseModel):
    query: str = Field(..., description="Search query string (keywords).")

class SemanticSearchInput(BaseModel):
    query: str = Field(..., description="Semantic/natural-language search query.")

class ListArchivesInput(BaseModel):
    earliest_date: Optional[date] = Field(
        default=None, description="Optional earliest date (YYYY-MM-DD)."
    )
    latest_date: Optional[date] = Field(
        default=None, description="Optional latest date (YYYY-MM-DD)."
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional tags to filter archives by."
    )

# ---------------- Thin wrappers (no reimplementation) ----------------

def _list_archives_tool(
    earliest_date: Optional[date] = None,
    latest_date: Optional[date] = None,
    tags: Optional[List[str]] = None,
) -> Any:
    # Forward named args exactly; your function handles optional params.
    return list_archives(
        earliest_date=earliest_date,
        latest_date=latest_date,
        tags=tags,
    )

# ---------------- Tools ----------------
TOOLS = [
    Tool.from_function(
        name="keyword_search",
        description=(
            "Keyword-based search over the archives. "
            "Use when you have specific terms, names, IDs, or exact phrases."
        ),
        func=keyword_search,
        args_schema=KeywordSearchInput,
    ),
    Tool.from_function(
        name="semantic_search",
        description=(
            "Semantic (natural-language) search over the archives. "
            "Use for fuzzy intent, paraphrases, or when you need conceptual recall."
        ),
        func=semantic_search,
        args_schema=SemanticSearchInput,
    ),
    StructuredTool.from_function(
        name="list_archives",
        description=(
            "List archives with optional filters. "
            "Parameters (all optional): earliest_date (YYYY-MM-DD), latest_date (YYYY-MM-DD), tags (list[str]). "
            "Use this to discover candidates/time windows before targeted searching."
        ),
        func=_list_archives_tool,
        args_schema=ListArchivesInput,
    ),
]

# ---------------- Prompt ----------------
SYSTEM_PROMPT = """You are a Retrieval-Augmented Generation (RAG) assistant.

Use the provided tools to search and read from the archive before answering.
Prefer high-signal results (direct matches, recent items, items with matching tags).
When both keyword and semantic search are applicable, try keyword for precision and semantic for coverage.
If relevant, use list_archives first to narrow by time window or tag.

Write concise answers, and include a short "Sources:" section with IDs/titles/links returned by the tools when possible.
Do NOT invent sources that were not returned by tools.
If no sources are retrieved, say so and clearly mark the response as best-effort.

Only include final conclusions in your reply. Keep your internal reasoning private.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        # With create_agent you don't add an explicit agent_scratchpad placeholder;
        # it manages tool-call turns internally.
    ]
)

# ---------------- Agent factory (returns a Runnable) ----------------
def build_rag_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    timeout: Optional[float] = None,
    debug: bool = False,
):
    llm = ChatOpenAI(model=model_name, temperature=temperature, timeout=timeout)
    # create_agent (LC 1.0.x) returns a Runnable that expects {"messages": [...]} and emits an AIMessage
    agent = create_agent(model=llm, tools=TOOLS, debug=debug)
    return agent

# ---------------- Helper for one-off queries ----------------
def answer_question(
    question: str,
    chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    **agent_kwargs,
) -> Dict[str, Any]:
    chat_history = chat_history or []
    agent = build_rag_agent(**agent_kwargs)

    # Render prompt to a list of chat messages, then pass as {"messages": ...}
    messages = PROMPT.format_messages(input=question, chat_history=chat_history)

    ai_msg = agent.invoke(messages)  # returns an AIMessage
    # Normalize the return to a dict similar to your previous shape
    return ai_msg


def answer_question_stream(
    question: str,
    chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    **agent_kwargs,
) -> Iterator[str]:
    """
    Streaming version of `answer_question`.

    Yields text chunks suitable for Flask streaming responses, e.g.:

        from flask import Flask, Response, stream_with_context
        app = Flask(__name__)

        @app.route("/ask")
        def ask():
            gen = answer_question_stream("Your question here")
            return Response(stream_with_context(gen), mimetype="text/plain")

    Requires `build_rag_agent` and `PROMPT` to be defined elsewhere in your module.
    """
    chat_history = chat_history or []
    agent = build_rag_agent(**agent_kwargs)

    # Render prompt into a list of messages that the agent expects.
    messages = PROMPT.format_messages(input=question, chat_history=chat_history)

    # Convert AIMessage chunks -> plain text chunks for easy streaming.
    parser = StrOutputParser()
    chain = agent

    for chunk in chain.stream({"messages": messages}):
        # `chunk` is a string segment; yield directly.
        if chunk:
            yield chunk


# ---------------- Example ----------------
if __name__ == "__main__":
    q = "Summarize discussions related to 'outstanding'."
    out = answer_question_stream(q)
    print("\n=== ANSWER ===\n")
    for kasi in out:
        print(kasi)