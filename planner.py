"""
The Planner: takes a broad research goal and breaks it into 3-5 concrete
sub-questions the Executor can answer one at a time.
"""

import os
import re
import google.generativeai as genai

MODEL = "gemini-3.1-flash-lite"

PLANNER_SYSTEM_PROMPT = (
    "You are a research planner. Given a research goal, break it down into "
    "3 to 5 specific, answerable sub-questions that together would cover "
    "the goal thoroughly. Output ONLY a numbered list, one sub-question per "
    "line, like:\n"
    "1. First sub-question\n"
    "2. Second sub-question\n"
    "No preamble, no explanation - just the numbered list."
)


def plan_research(goal: str) -> list[str]:
    """Returns a list of sub-question strings for the given research goal."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=PLANNER_SYSTEM_PROMPT,
    )

    response = model.generate_content(goal)
    text = response.text.strip()

    sub_questions = []
    for line in text.split("\n"):
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            sub_questions.append(match.group(1).strip())

    if not sub_questions:
        sub_questions = [goal]

    return sub_questions