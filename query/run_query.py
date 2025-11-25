from services.factory import get_embedding_service, get_vector_store
from google import genai


def synthesize_with_gemini(prompt: str) -> str:
    client = genai.Client()
    resp = client.responses.create(model="gemini-2.5-flash-lite", input=prompt, max_output_tokens=512)
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    try:
        return resp.output[0].content[0].text
    except Exception:
        return str(resp)


def search_and_synthesize(query: str, n_results=3):
    embedder = get_embedding_service()
    store = get_vector_store()
    q_emb = embedder.embed([query])[0]
    hits = store.query(q_emb, n_results=n_results)
    # Format passages for prompt
    docs = hits.get("documents", [[]])[0]
    metas = hits.get("metadatas", [[]])[0]
    passages = "\n\n".join([f"[{i + 1}] {m.get('source', '')}: {d}" for i, (d, m) in enumerate(zip(docs, metas))])
    prompt = f"Answer using only the passages. Query:\n{query}\nPassages:\n{passages}\nIf not found, say 'Answer not found.'"
    return synthesize_with_gemini(prompt), hits


if __name__ == "__main__":
    q = input("Question: ").strip()
    ans, hits = search_and_synthesize(q)
    print(ans)
    for i, h in enumerate(hits.get("documents", [[]])[0], 1):
        print(f"[{i}] {h[:200]}")
