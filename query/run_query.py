import logging

from services.factory import get_embedding_service, get_vector_store, get_llm_service


def search_and_synthesize(query: str, n_results=3, cfg=None):
    logging.info(f'synthesizing query: {query}')

    embedder = get_embedding_service(cfg)
    store = get_vector_store(cfg)
    q_emb = embedder.embed([query])[0]
    hits = store.query(q_emb, n_results=n_results)
    # Format passages for prompt
    docs = hits.get("documents", [[]])[0]
    metas = hits.get("metadatas", [[]])[0]
    passages = "\n\n".join([f"[{i + 1}] {m.get('source', '')}: {d}" for i, (d, m) in enumerate(zip(docs, metas))])
    prompt = f"""
    Answer using only the passages. 
    Query:
    {query}
    
    Passages:
    {passages}
    
    If not found, say 'Answer not found.
    """

    llm = get_llm_service()
    return llm.synthesize(prompt), hits


if __name__ == "__main__":
    q = input("Question: ").strip()
    ans, hits = search_and_synthesize(q)
    print(ans)
    for i, h in enumerate(hits.get("documents", [[]])[0], 1):
        print(f"[{i}] {h[:200]}")
