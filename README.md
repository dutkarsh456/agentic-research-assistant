# Agentic Research Assistant (Work in Progress)

A multi-agent AI system that autonomously plans, researches, and reports on a
given topic using tool-use, RAG-based memory, and a self-critique loop.

> **Status:** Week 1 - single agent with tool use (calculator, web search).
> Planner/Executor/Critic architecture coming in Week 2-3.

## Why this project

Most "AI projects" are a single LLM call wrapped in a chatbot UI. This one
implements the actual patterns used in production agent systems:
tool-calling loops, multi-step planning, and self-review - the same ideas
behind LangGraph, AutoGPT, and modern agentic tool use.

## Architecture (Week 1)
The loop is called **ReAct** (Reason + Act): the LLM decides whether to
answer directly or call a tool, Python executes the tool, the result goes
back to the LLM, and this repeats until a final answer is ready. Gemini's
SDK handles this loop automatically via `enable_automatic_function_calling`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# create a .env file with:
# GEMINI_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
```

Both API keys are free:
- **Gemini API key** (no credit card needed): https://aistudio.google.com/apikey
- **Tavily API key** (free tier, no credit card needed): https://tavily.com

## Run

```bash
python agent.py "Research the latest trends in agentic AI and summarize in 3 bullet points"
```

## Project structure
## Roadmap

- [x] Week 1: Single agent + tool use (calculator, web search)
- [ ] Week 2: Planner agent (breaks goal into steps) + vector memory (ChromaDB)
-  [x] Week 3: Executor + Critic agents, self-revision loop
- [ ] Week 4: Eval harness, polish, architecture diagram, demo

## Tech Stack

Python, Google Gemini API (`gemini-3.5-flash`, free tier, function calling),
Tavily (web search), LangGraph (from Week 2), ChromaDB (from Week 2)

## Notes / learnings

- Google's free tier quota varies a lot by model - newer models (`gemini-3.5-flash`,
  `gemini-3.1-flash-lite`) had quota available on a fresh account; older
  models (`gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-2.5-flash`) returned
  either `RESOURCE_EXHAUSTED` or `404 no longer available to new users`.
  Check https://aistudio.google.com/rate-limit for current per-model limits.
- `.env` and `venv/` are gitignored - never commit API keys.
XX