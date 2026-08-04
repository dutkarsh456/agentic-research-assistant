"""
Week 1 MVP: Single Agent with Tool Use (Gemini free API).

The core loop here - called "ReAct" (Reason + Act) - is the foundation
EVERY agent framework (LangGraph, AutoGPT, CrewAI, etc.) is built on:

    1. Send the user's goal + available tools to the LLM
    2. LLM decides: answer directly, OR call a tool
    3. If it calls a tool -> we execute it in Python -> feed result back to LLM
    4. Repeat until LLM gives a final answer (no more tool calls)

Gemini's SDK can handle steps 2-4 automatically ("automatic function
calling") when we enable it on a chat session - it calls our Python
functions directly and loops until it has a final answer.
"""

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

from tools.tools import AVAILABLE_TOOLS

load_dotenv()

MODEL = "gemini-3.5-flash" # free tier, fast, supports function calling

SYSTEM_PROMPT = (
    "You are a research assistant agent. You have access to tools "
    "(calculator, web_search). Use them whenever they would improve "
    "the accuracy of your answer - do not guess at facts you can look up. "
    "Once you have enough information, give a clear, well-structured final answer."
)


def run_agent(user_goal: str) -> str:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel(
        model_name=MODEL,
        tools=AVAILABLE_TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )

    chat = model.start_chat(enable_automatic_function_calling=True)

    response = chat.send_message(user_goal)
    return response.text


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY in your .env file first.")
        sys.exit(1)

    goal = " ".join(sys.argv[1:]) or "What is 47 * 89, and who won the last IPL final?"
    print(f"Goal: {goal}\n")

    answer = run_agent(goal)
    print("=== FINAL ANSWER ===")
    print(answer)