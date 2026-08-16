"""
The Executor: a single agent with tool use (calculator, web_search).

Same ReAct pattern from Week 1. In Week 3, it can optionally accept
feedback from the Critic to improve a previous weak answer.
"""

import os
import google.generativeai as genai

from tools.tools import AVAILABLE_TOOLS

MODEL = "gemini-3.1-flash-lite"

EXECUTOR_SYSTEM_PROMPT = (
    "You are a research assistant. You have access to tools "
    "(calculator, web_search). Use them whenever they would improve "
    "the accuracy of your answer - do not guess at facts you can look up. "
    "Answer the question directly and concisely, citing sources by name "
    "when you use web_search."
)


def run_agent(question: str, feedback: str | None = None) -> str:
    """Answers a single question, using tools as needed.

    If feedback is given (from a previous failed Critic review), the
    question is reframed to explicitly address that feedback.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel(
        model_name=MODEL,
        tools=AVAILABLE_TOOLS,
        system_instruction=EXECUTOR_SYSTEM_PROMPT,
    )

    chat = model.start_chat(enable_automatic_function_calling=True)

    if feedback:
        prompt = (
            f"{question}\n\n"
            f"Your previous answer to this was reviewed and needs improvement: "
            f"{feedback}\n"
            f"Please provide a better, more specific answer."
        )
    else:
        prompt = question

    response = chat.send_message(prompt)
    return response.text