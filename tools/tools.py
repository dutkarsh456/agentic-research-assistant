"""
Tools the agent is allowed to use.

Gemini's SDK supports "automatic function calling": if a Python function has
clear type hints and a docstring, we can hand the function itself to the
model and it figures out the schema automatically.
"""

import os
import requests


def calculator(expression: str) -> str:
    """Evaluates a basic math expression and returns the numeric result.

    Use this whenever the task requires arithmetic instead of trying to
    compute it yourself.

    Args:
        expression: A math expression, e.g. "23 * 47 + 100"
    """
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: expression contains disallowed characters."
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def web_search(query: str) -> str:
    """Searches the live web for current, factual, or up-to-date information.

    Use this whenever the task requires information you cannot be certain
    about from memory alone, especially anything recent.

    Args:
        query: The search query
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set in .env file."

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 5,
                "include_answer": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        output = []
        if data.get("answer"):
            output.append(f"Quick answer: {data['answer']}")

        for i, result in enumerate(data.get("results", []), 1):
            output.append(
                f"\n[Source {i}] {result.get('title', 'Untitled')}\n"
                f"URL: {result.get('url', '')}\n"
                f"Content: {result.get('content', '')[:500]}"
            )

        return "\n".join(output) if output else "No results found."

    except Exception as e:
        return f"Error during web search: {e}"


AVAILABLE_TOOLS = [calculator, web_search]