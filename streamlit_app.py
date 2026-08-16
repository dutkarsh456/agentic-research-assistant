"""
Streamlit web UI for the Agentic Research Assistant.

Wraps the Planner -> Executor -> Critic -> Memory pipeline (research.py)
in a simple browser interface so anyone can try it without touching
the command line.

Run locally with: streamlit run streamlit_app.py
"""

import streamlit as st
from planner import plan_research
from agent import run_agent
from critic import critique_answer
from memory.memory import store_research, search_memory
from research import synthesize_report

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔎")

st.title("Agentic Research Assistant")
st.caption(
    "A Planner -> Executor -> Critic multi-agent pipeline with RAG-based "
    "memory. Enter a research goal below and watch each step run live."
)

goal = st.text_input(
    "Research goal",
    placeholder="e.g. What are the biggest trends in agentic AI in 2026?",
)

run_button = st.button("Run research", type="primary")

if run_button and goal.strip():
    log = st.empty()
    logs = []

    def add_log(msg: str):
        logs.append(msg)
        log.markdown("\n\n".join(logs))

    with st.spinner("Running the pipeline - this can take a few minutes on the free tier..."):
        add_log("**[1/6] Checking memory for related past research...**")
        past_findings = search_memory(goal, n_results=3)
        if past_findings:
            add_log(f"Found {len(past_findings)} related entries from memory.")
        else:
            add_log("No related past research found - starting fresh.")

        add_log("**[2/6] Planning sub-questions...**")
        try:
            sub_questions = plan_research(goal)
        except Exception as e:
            st.error(f"Planning failed (likely a free-tier rate limit): {e}")
            st.stop()

        for i, q in enumerate(sub_questions, 1):
            add_log(f"{i}. {q}")

        add_log("**[3-4/6] Researching + critiquing each sub-question...**")
        qa_pairs = []
        for i, question in enumerate(sub_questions, 1):
            add_log(f"[{i}/{len(sub_questions)}] {question}")
            try:
                answer = run_agent(question)
                passed, feedback = critique_answer(question, answer)
                add_log(f"Critic: {'PASS' if passed else 'FAIL'} - {feedback}")
                if not passed:
                    add_log("Redoing with feedback...")
                    answer = run_agent(question, feedback=feedback)
            except Exception as e:
                st.error(f"Research step failed (likely a free-tier rate limit): {e}")
                st.stop()

            qa_pairs.append((question, answer))
            store_research(question, answer)

        add_log("**[5/6] Saved all findings to memory.**")
        add_log("**[6/6] Writing final report...**")

        try:
            report = synthesize_report(goal, qa_pairs)
        except Exception as e:
            st.error(f"Report synthesis failed (likely a free-tier rate limit): {e}")
            st.stop()

    st.success("Done!")
    st.markdown("## Final Report")
    st.markdown(report)

elif run_button:
    st.warning("Please enter a research goal first.")

st.divider()
st.caption(
    "Built with Google Gemini (free tier), Tavily search, and ChromaDB. "
    "Free-tier rate limits mean this may be slow or occasionally fail - "
    "that's expected, not a bug. See the GitHub repo for the full "
    "architecture and eval harness."
)
