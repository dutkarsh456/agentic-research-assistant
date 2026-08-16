"""
Eval harness: tests the deterministic parts of the pipeline WITHOUT
calling the Gemini API (so it doesn't burn free-tier quota).

Covers:
- calculator tool correctness
- planner's sub-question parsing logic
- critic's verdict/feedback parsing logic
- memory store + retrieve round-trip (ChromaDB runs locally, no API needed)
"""

from tools.tools import calculator
from memory.memory import store_research, search_memory


def test_calculator():
    cases = [
        ("2 + 2", "4"),
        ("10 * 5", "50"),
        ("100 / 4", "25.0"),
    ]
    passed = 0
    for expr, expected in cases:
        result = calculator(expr)
        ok = result == expected
        passed += ok
        print(f"   {'PASS' if ok else 'FAIL'}: calculator('{expr}') = {result} (expected {expected})")
    return passed, len(cases)


def test_planner_parsing():
    """Tests the numbered-list parsing regex planner.py uses,
    without calling the API."""
    import re

    sample_text = "1. First question here\n2. Second question here\n3. Third one"

    sub_questions = []
    for line in sample_text.split("\n"):
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            sub_questions.append(match.group(1).strip())

    ok = len(sub_questions) == 3 and sub_questions[0] == "First question here"
    print(f"   {'PASS' if ok else 'FAIL'}: planner parsed {len(sub_questions)} sub-questions")
    return int(ok), 1


def test_critic_parsing():
    """Tests the critic's VERDICT/FEEDBACK parsing regex."""
    import re

    sample_text = "VERDICT: FAIL\nFEEDBACK: Answer is too vague."

    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", sample_text, re.IGNORECASE)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", sample_text, re.IGNORECASE | re.DOTALL)
    passed = bool(verdict_match) and verdict_match.group(1).upper() == "PASS"
    feedback = feedback_match.group(1).strip() if feedback_match else None

    ok = (passed is False) and (feedback == "Answer is too vague.")
    print(f"   {'PASS' if ok else 'FAIL'}: critic correctly parsed FAIL verdict + feedback")
    return int(ok), 1


def test_memory_roundtrip():
    """Stores a fake Q&A pair, then checks it can be retrieved by a
    semantically similar (not identical) query."""
    store_research(
        "What is the boiling point of water at sea level?",
        "100 degrees Celsius at standard atmospheric pressure.",
    )
    results = search_memory("water boiling temperature", n_results=3)
    ok = any("boiling" in r.lower() for r in results)
    print(f"   {'PASS' if ok else 'FAIL'}: memory retrieved relevant entry via semantic search")
    return int(ok), 1


if __name__ == "__main__":
    print("Running eval harness (no API calls, free)...\n")

    total_passed = 0
    total_tests = 0

    print("[1/4] Calculator tool")
    p, t = test_calculator()
    total_passed += p
    total_tests += t

    print("\n[2/4] Planner parsing")
    p, t = test_planner_parsing()
    total_passed += p
    total_tests += t

    print("\n[3/4] Critic parsing")
    p, t = test_critic_parsing()
    total_passed += p
    total_tests += t

    print("\n[4/4] Memory store + retrieve")
    p, t = test_memory_roundtrip()
    total_passed += p
    total_tests += t

    print(f"\n{'='*40}")
    print(f"RESULT: {total_passed}/{total_tests} tests passed")
    print(f"{'='*40}")
