"""AI Journalist: multilingual LangGraph workflow (plan -> research -> write -> critique -> revise)."""

import re
from typing import Callable, List, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

MAX_REVISIONS = 2


class WriterState(TypedDict, total=False):
    topic: str
    language: str
    content_type: str
    tone: str
    length: str
    use_web: bool
    queries: List[str]
    sources: List[dict]
    draft: str
    feedback: str
    revisions: int


def _ask(llm, system: str, user: str) -> str:
    text = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
    # Some reasoning models return their chain of thought inside <think> tags; drop it.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def build_graph(llm, search_fn: Callable[[str], List[dict]]):
    """llm: any LangChain chat model. search_fn: query -> [{'title','url','content'}]."""

    def plan(state: WriterState):
        if not state.get("use_web"):
            return {"queries": []}
        text = _ask(
            llm,
            "You generate web search queries. Reply with exactly 3 short English queries, one per line, no numbering or extra text.",
            f"Topic: {state['topic']}\nContent type: {state['content_type']}",
        )
        queries = [re.sub(r"^[\s\-\*\d\.\)]+", "", ln).strip() for ln in text.splitlines()]
        return {"queries": [q for q in queries if q][:3]}

    def research(state: WriterState):
        seen, sources = set(), []
        for q in state.get("queries", []):
            try:
                for r in search_fn(q):
                    if r["url"] not in seen:
                        seen.add(r["url"])
                        sources.append(r)
            except Exception:
                continue  # one failed search should not kill the run
        return {"sources": sources[:6]}

    def write(state: WriterState):
        sources = state.get("sources", [])
        context = "\n\n".join(
            f"[{i}] {s['title']} ({s['url']})\n{s['content']}" for i, s in enumerate(sources, 1)
        ) or "No sources available."
        system = (
            f"You are a skilled writer. Write a {state['content_type']} in {state['language']} "
            f"with a {state['tone']} tone, {state['length']}. "
            f"The entire output must be in {state['language']} (proper nouns excepted). "
            "Use the provided sources and cite them inline like [1], [2]. "
            "Never invent statistics, quotes or facts. If sources are missing, write from general knowledge and stay cautious."
        )
        draft = _ask(llm, system, f"Topic: {state['topic']}\n\nSources:\n{context}")
        return {"draft": draft, "revisions": 0}

    def critique(state: WriterState):
        system = (
            "You are a strict editor. Check the draft for: (1) written fully in "
            f"{state['language']}, (2) {state['tone']} tone, (3) length: {state['length']}, "
            "(4) clear structure, (5) claims not supported by sources. "
            "If it is good enough, reply with exactly: APPROVED. "
            "Otherwise reply with a short bullet list of concrete fixes."
        )
        return {"feedback": _ask(llm, system, state["draft"])}

    def revise(state: WriterState):
        system = (
            f"You are a skilled editor. Rewrite the draft in {state['language']} applying the feedback. "
            "Return only the improved piece."
        )
        draft = _ask(llm, system, f"Draft:\n{state['draft']}\n\nFeedback:\n{state['feedback']}")
        return {"draft": draft, "revisions": state.get("revisions", 0) + 1}

    def after_plan(state: WriterState):
        return "research" if state.get("use_web") else "write"

    def after_critique(state: WriterState):
        approved = state["feedback"].strip().upper().startswith("APPROVED")
        return END if approved or state.get("revisions", 0) >= MAX_REVISIONS else "revise"

    g = StateGraph(WriterState)
    for name, fn in [("plan", plan), ("research", research), ("write", write),
                     ("critique", critique), ("revise", revise)]:
        g.add_node(name, fn)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", after_plan, {"research": "research", "write": "write"})
    g.add_edge("research", "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", after_critique, {"revise": "revise", END: END})
    g.add_edge("revise", "critique")
    return g.compile()