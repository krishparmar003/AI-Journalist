"""FastAPI backend for AI Journalist, a multilingual research-and-writing agent (Vercel entrypoint)."""

import hmac
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tavily import TavilyClient

from agent import build_graph

load_dotenv()

LANGUAGES = {"English", "Hindi", "Gujarati", "Marathi", "Bengali", "Tamil"}
CONTENT_TYPES = {"Article", "Blog post", "LinkedIn post", "Essay", "Email"}
TONES = {"Professional", "Conversational", "Formal", "Inspirational", "Analytical"}
LENGTHS = {"short": "around 300 words", "medium": "around 800 words", "long": "around 1500 words"}
MODELS = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant"}

# Two ways to use the app:
#  1. Visitors bring their own Groq/Tavily keys (sent with the request, never stored).
#  2. The owner enters OWNER_PASSWORD; the server then uses its own GROQ_API_KEY / TAVILY_API_KEY
#     from environment variables. Server keys are never used without the correct password.

app = FastAPI(title="AI Journalist")


class GenerateRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    language: str
    content_type: str
    tone: str
    length: str = "medium"
    model: str = "llama-3.3-70b-versatile"
    use_web: bool = True
    groq_key: Optional[str] = None
    tavily_key: Optional[str] = None
    owner_password: Optional[str] = Field(default=None, max_length=200)


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "429" in low or "rate limit" in low:
        return "Groq rate limit reached. Wait a minute, or switch to the 8B model."
    if "401" in low or "invalid api key" in low or "unauthorized" in low:
        return "Invalid API key (check your Groq and Tavily keys)."
    return f"Something went wrong: {msg[:300]}"


def resolve_keys(req: GenerateRequest):
    """Return (groq_key, tavily_key). Server keys are only released for the correct owner password."""
    if req.owner_password:
        expected = os.getenv("OWNER_PASSWORD", "")
        if not expected:
            raise HTTPException(403, "Owner access is not configured on this deployment.")
        if not hmac.compare_digest(req.owner_password.encode(), expected.encode()):
            time.sleep(1.5)  # slow down password guessing
            raise HTTPException(401, "Incorrect owner password.")
        return os.getenv("GROQ_API_KEY"), os.getenv("TAVILY_API_KEY")
    return req.groq_key, req.tavily_key


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.language not in LANGUAGES:
        raise HTTPException(422, "Unsupported language.")
    if req.content_type not in CONTENT_TYPES:
        raise HTTPException(422, "Unsupported content type.")
    if req.tone not in TONES:
        raise HTTPException(422, "Unsupported tone.")
    if req.length not in LENGTHS:
        raise HTTPException(422, "Unsupported length.")
    if req.model not in MODELS:
        raise HTTPException(422, "Unsupported model.")

    groq_key, tavily_key = resolve_keys(req)
    if not groq_key:
        raise HTTPException(400, "A Groq API key is required.")
    if req.use_web and not tavily_key:
        raise HTTPException(400, "A Tavily API key is required for web research (or turn web research off).")

    def events():
        try:
            llm = ChatGroq(model=req.model, api_key=groq_key, temperature=0.6)
            tavily = TavilyClient(api_key=tavily_key) if req.use_web else None

            def search(query: str):
                res = tavily.search(query=query, max_results=3)
                return [
                    {"title": r.get("title", "Untitled"), "url": r["url"], "content": r.get("content", "")[:800]}
                    for r in res.get("results", [])
                ]

            graph = build_graph(llm, search)
            state = {
                "topic": req.topic.strip(),
                "language": req.language,
                "content_type": req.content_type,
                "tone": req.tone,
                "length": LENGTHS[req.length],
                "use_web": req.use_web,
            }
            for update in graph.stream(state, stream_mode="updates"):
                for node, out in update.items():
                    state.update(out)
                    yield sse({"type": "step", "node": node})
            yield sse({
                "type": "result",
                "draft": state.get("draft", ""),
                "sources": state.get("sources", []),
                "feedback": state.get("feedback", ""),
                "revisions": state.get("revisions", 0),
            })
        except Exception as e:  # surfaced to the UI instead of a broken stream
            yield sse({"type": "error", "message": friendly_error(e)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Local dev only: on Vercel, files in public/ are served by the CDN.
if not os.getenv("VERCEL"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")