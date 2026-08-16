"""
Memory: stores past question/answer pairs in a vector database (ChromaDB)
so the agent can recall related research from earlier runs.

STORE: every answered sub-question gets saved with an embedding (a
numeric fingerprint of its meaning).
RETRIEVE: before researching something new, we search for past entries
with a similar meaning and feed the most relevant ones back to the agent.
"""

import chromadb

# Persistent means data survives between runs (saved to disk).
_client = chromadb.PersistentClient(path="memory/chroma_data")
_collection = _client.get_or_create_collection(name="research_memory")


def store_research(question: str, answer: str) -> None:
    """Saves a question/answer pair to memory."""
    entry_id = str(_collection.count())
    _collection.add(
        documents=[f"Q: {question}\nA: {answer}"],
        ids=[entry_id],
    )


def search_memory(query: str, n_results: int = 3) -> list[str]:
    """Returns up to n_results past Q&A entries most relevant to the query."""
    count = _collection.count()
    if count == 0:
        return []

    results = _collection.query(
        query_texts=[query],
        n_results=min(n_results, count),
    )
    return results["documents"][0] if results["documents"] else []