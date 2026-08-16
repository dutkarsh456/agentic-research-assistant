# Agentic Research Assistant

A multi-agent AI system that autonomously plans, researches, critiques,
and reports on a given topic using tool-use, RAG-based memory, and a
self-correction loop.

> **Status:** All 4 weeks complete - Planner, Executor, Critic, memory,
> and an eval harness.

## Why this project

Most "AI projects" are a single LLM call wrapped in a chatbot UI. This
one implements the actual patterns used in production agent systems:
tool-calling loops, multi-step planning, self-critique, and long-term
memory - the same ideas behind LangGraph, AutoGPT, and modern agentic
tool use.

## Architecture
This is the **Planner -> Executor -> Critic** pattern: instead of one
big vague LLM call, the goal is decomposed into small focused steps,
each answer is independently reviewed, and everything is persisted so
future runs can build on past research (RAG).

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
python research.py "What are the biggest trends in agentic AI in 2026?"
```

## Run the eval harness

Tests the deterministic parts of the pipeline (tool correctness, parsing
logic, memory retrieval) **without calling the Gemini API**, so it runs
instantly and never hits rate limits:

```bash
python eval.py
```

## Project structure
## Roadmap

- [x] Week 1: Single agent + tool use (calculator, web search)
- [x] Week 2: Planner agent + vector memory (ChromaDB)
- [x] Week 3: Executor + Critic agents, self-revision loop
- [x] Week 4: Eval harness, polish, architecture diagram

## Tech Stack

Python, Google Gemini API (`gemini-3.1-flash-lite`, free tier, function
calling), Tavily (web search), ChromaDB (local vector memory, no
external API needed)

## Notes / learnings

- Google's free tier quota varies a lot by model and resets daily. Check
  https://aistudio.google.com/rate-limit for current per-model limits
  before choosing a model.
- The pipeline sleeps between API calls to stay under the free-tier
  rate limit (5 requests/minute on some models).
- `.env` and `venv/` are gitignored - never commit API keys.
- The eval harness deliberately avoids live API calls so it can run in
  CI or a demo without burning quota or needing secrets.
