import logging

from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator, EvaluationResult

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
    Add references.
    Query:
    {query}
    
    Passages:
    {passages}
    
    If not found, say 'Answer not found.
    """

    llm = get_llm_service()
    response = llm.synthesize(prompt)
    response_text = response if isinstance(response, str) else getattr(response, "text", str(response))

    # --- contexts for evaluators ---
    contexts = [f"{m.get('source', 'source')}: {d}" for d, m in zip(docs, metas)]

    # --- evaluators (only 2, as requested) ---
    evaluations = {}

    try:
        rel_eval = AnswerRelevancyEvaluator()
        rel_res: EvaluationResult = rel_eval.evaluate(query=query, response=response_text, contexts=contexts)
        evaluations["AnswerRelevancyEvaluator"] = rel_res.to_dict() if hasattr(rel_res, "to_dict") else rel_res
        logging.info(f"AnswerRelevancyEvaluator -> {evaluations['AnswerRelevancyEvaluator']}")
    except Exception as e:
        evaluations["AnswerRelevancyEvaluator"] = {"error": str(e)}
        logging.debug(f"AnswerRelevancyEvaluator failed: {e}")

    try:
        faith_eval = FaithfulnessEvaluator()
        faith_res: EvaluationResult = faith_eval.evaluate(query=query, response=response_text, contexts=contexts)
        evaluations["FaithfulnessEvaluator"] = faith_res.to_dict() if hasattr(faith_res, "to_dict") else faith_res
        logging.info(f"FaithfulnessEvaluator -> {evaluations['FaithfulnessEvaluator']}")
    except Exception as e:
        evaluations["FaithfulnessEvaluator"] = {"error": str(e)}
        logging.debug(f"FaithfulnessEvaluator failed: {e}")

    return response, hits


async def synthesize(query: str, n_results=3, cfg=None):
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
    answer = await llm.synthesize_agentic(prompt)
    return answer, hits


if __name__ == "__main__":
    q = input("Question: ").strip()
    ans, hits = search_and_synthesize(q)
    print(ans)
    for i, h in enumerate(hits.get("documents", [[]])[0], 1):
        print(f"[{i}] {h[:200]}")
