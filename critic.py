"""
The Critic: reviews an Executor's answer to a sub-question and decides
whether it's good enough to keep, or needs to be redone.
"""

import os
import re
import google.generativeai as genai

MODEL = "gemini-3.1-flash-lite"

CRITIC_SYSTEM_PROMPT = (
    "You are a strict research quality reviewer. You will be given a "
    "sub-question and an answer to it. Judge whether the answer is "
    "specific, accurate-sounding, and actually addresses the question "
    "(vague answers like 'it depends' or answers that dodge the question "
    "should FAIL). Respond in EXACTLY this format, nothing else:\n"
    "VERDICT: PASS or FAIL\n"
    "FEEDBACK: one sentence explaining what's missing or wrong (or 'Good "
    "answer.' if PASS)"
)


def critique_answer(question: str, answer: str) -> tuple[bool, str]:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=CRITIC_SYSTEM_PROMPT,
    )

    prompt = f"Question: {question}\n\nAnswer: {answer}"
    response = model.generate_content(prompt)
    text = response.text.strip()

    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    passed = bool(verdict_match) and verdict_match.group(1).upper() == "PASS"
    feedback = feedback_match.group(1).strip() if feedback_match else "No feedback given."

    return passed, feedback
