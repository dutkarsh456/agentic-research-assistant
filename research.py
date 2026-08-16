"""
Week 2: Planner -> Executor pipeline with vector memory.

Flow for one research goal:
    1. MEMORY CHECK   - search past research for anything related
    2. PLAN           - break the goal into 3-5 sub-questions
    3. EXECUTE        - answer each sub-question one at a time (Week 1 agent)
    4. REMEMBER       - save each Q&A pair to memory for next time
    5. SYNTHESIZE     - combine all answers into one final report

NOTE: free-tier Gemini quota is only 5 requests/minute, so we sleep
between calls to avoid hitting RESOURCE_EXHAUSTED errors.
"""

import os
import sys
import time
from dotenv import load_dotenv
import google.generativeai as genai

from planner import plan_research
from agent import run_agent
from memory.memory import store_research, search_memory

load_dotenv()

MODEL = "gemini-3.1-flash-lite"
DELAY_SECONDS = 15  # free tier = 5 req/min, so ~12s minimum between calls

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a research assistant writing a final report. You will be "
    "given a research goal and a set of sub-question/answer pairs. "
    "Synthesize them into a clear, well-structured report with a short "
    "intro, organized findings (use headers/bullets), and a brief "
    "conclusion. Do not just list the Q&A pairs verbatim - write it as a "
    "cohesive report."
)


def synthesize_report(goal: str, qa_pairs: list[tuple[str, str]]) -> str:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYNTHESIS_SYSTEM_PROMPT,
    )

    qa_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
    prompt = f"Research goal: {goal}\n\nSub-question findings:\n\n{qa_text}"

    response = model.generate_content(prompt)
    return response.text


def research(goal: str) -> str:
    print(f"\n{'='*60}\nGOAL: {goal}\n{'='*60}")

    print("\n[1/5] Checking memory for related past research...")
    past_findings = search_memory(goal, n_results=3)
    if past_findings:
        print(f"   Found {len(past_findings)} related entries from memory.")
    else:
        print("   No related past research found - starting fresh.")

    print("\n[2/5] Planning sub-questions...")
    sub_questions = plan_research(goal)
    for i, q in enumerate(sub_questions, 1):
        print(f"   {i}. {q}")

    print(f"\n   Waiting {DELAY_SECONDS}s to respect free-tier rate limit...")
    time.sleep(DELAY_SECONDS)

    print("\n[3/5] Researching each sub-question...")
    qa_pairs = []
    for i, question in enumerate(sub_questions, 1):
        print(f"   Researching {i}/{len(sub_questions)}: {question}")
        answer = run_agent(question)
        qa_pairs.append((question, answer))
        store_research(question, answer)

        if i < len(sub_questions):
            print(f"   Waiting {DELAY_SECONDS}s to respect free-tier rate limit...")
            time.sleep(DELAY_SECONDS)

    print("\n[4/5] Saved all findings to memory.")

    print(f"\n   Waiting {DELAY_SECONDS}s to respect free-tier rate limit...")
    time.sleep(DELAY_SECONDS)

    print("\n[5/5] Writing final report...\n")
    report = synthesize_report(goal, qa_pairs)
    return report


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY in your .env file first.")
        sys.exit(1)

    goal = " ".join(sys.argv[1:]) or "What are the latest trends in agentic AI?"
    final_report = research(goal)

    print("=== FINAL REPORT ===\n")
    print(final_report)
