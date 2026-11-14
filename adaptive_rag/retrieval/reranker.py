import torch
from sentence_transformers import CrossEncoder
from adaptive_rag.config import RERANKING_MODEL

def get_reranker(model: str = RERANKING_MODEL) -> CrossEncoder:
    """
        Load the CrossEncoder model used for passage reranking.

        Args:
            model: Name or path of the CrossEncoder model.

        Returns:
            The loaded CrossEncoder model.
    """
    return CrossEncoder(model, activation_fn=torch.nn.Sigmoid())

def rerank(
        reranker: CrossEncoder,
        query: str,
        candidate_docs: list[dict[str, str]],
        rerank_top_k: int
) -> list[tuple[dict[str, str], float]]:
    """
        Rerank candidate passages according to their relevance to a query.

        Args:
            reranker: CrossEncoder used to score query-document pairs.
            query: Query against which candidate passages are scored.
            candidate_docs: Candidate passages containing a text field.
            rerank_top_k: Maximum number of highest-scoring passages to return.

        Returns:
            A list of (document, score) tuples sorted by decreasing relevance.
    """
    if not candidate_docs:
        return []

    if rerank_top_k <= 0:
        raise ValueError("top_k must be positive")

    query_doc_pairs = [[query, f"{doc['title']}\n{doc['text']}"] for doc in candidate_docs]
    scores = reranker.predict(query_doc_pairs)

    scored_results = sorted(
        (
            (doc, float(score)) for doc, score in zip(candidate_docs, scores)
        ),
        key=lambda x: x[1],
        reverse=True
    )
    return scored_results[:rerank_top_k]