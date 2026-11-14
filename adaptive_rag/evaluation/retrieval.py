from math import log2

def recall_at_k(
    gold_passage_ids: list[str],
    retrieved_docs: list[dict[str, str]],
    k: int
) -> float:
    """Calculate passage-level Recall@K using exact UUID matches."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    gold_ids = set(gold_passage_ids)
    if not gold_ids:
        return 0.0

    retrieved_ids = {doc["uuid"] for doc in retrieved_docs[:k]}

    return len(gold_ids & retrieved_ids) / len(gold_ids)


def avg_recall_at_k(recalls: list[float]) -> float:
    """
        Calculate average Recall@K across multiple questions.

        Args:
            recalls: List of per-question Recall@K scores.

        Returns:
            Mean Recall@K across all evaluated questions.
    """
    if not recalls:
        raise ValueError("recalls must not be empty")
    return sum(recalls) / len(recalls)


def ndcg_at_k(
    gold_passage_ids: list[str],
    retrieved_docs: list[dict[str, str]],
    k: int
) -> float:
    """Calculate passage-level NDCG@K with binary UUID relevance."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    gold_ids = set(gold_passage_ids)
    if not gold_ids:
        return 0.0

    seen = set()
    dcg = 0.0

    for i, doc in enumerate(retrieved_docs[:k]):
        passage_id = doc["uuid"]

        if passage_id in gold_ids and passage_id not in seen:
            dcg += 1 / log2(i + 2)
            seen.add(passage_id)

    idcg = sum(
        1 / log2(i + 2)
        for i in range(min(k, len(gold_ids)))
    )

    return dcg / idcg


def avg_ndcg_at_k(ndcgs: list[float]) -> float:
    """
        Calculate average NDCG@K across multiple questions.

        Args:
            ndcgs: List of per-question NDCG@K scores.

        Returns:
            Mean NDCG@K across all evaluated questions.
    """
    if not ndcgs:
        raise ValueError("NDCGs must not be empty")

    return sum(ndcgs) / len(ndcgs)