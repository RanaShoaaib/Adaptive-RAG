import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder
from adaptive_rag.embedding.embedder import embed_queries
from adaptive_rag.config import COLLECTION_NAME, HYBRID_SEARCH_TOP_K, HYBRID_SEARCH_ALPHA, FINAL_CONTEXT_TOP_K, SUBQUERY_TOP_K, RELEVANCE_THRESHOLD
from adaptive_rag.retrieval.reranker import rerank
from adaptive_rag.planner.prompted import rewrite_query
from transformers import AutoTokenizer, AutoModelForCausalLM


def hybrid_search(
        client: weaviate.WeaviateClient,
        model: SentenceTransformer,
        query: dict[str, str],
        collection_name: str = COLLECTION_NAME,
        alpha: float = HYBRID_SEARCH_ALPHA,
        hybrid_top_k: int = HYBRID_SEARCH_TOP_K
) -> list[dict[str, str]]:
    """
        Retrieve candidate passages using hybrid lexical and semantic search.

        BM25 search uses the lexical query while vector search uses an embedding
        of the semantic query. The two signals are combined by Weaviate.

        Args:
            client: Connected Weaviate client.
            model: SentenceTransformer used to embed the semantic query.
            query: Dictionary containing lexical_query and semantic_query.
            collection_name: Name of the Weaviate collection to search.
            alpha: Weight controlling the lexical-semantic hybrid search balance.
            hybrid_top_k: Maximum number of candidate passages to retrieve.

        Returns:
            Retrieved passages containing UUID, title, and text.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if hybrid_top_k <= 0:
        raise ValueError("hybrid_top_k must be positive")

    collection = client.collections.get(collection_name)
    query_vector = embed_queries(model, [query["semantic_query"]])[0]

    response = collection.query.hybrid(
        query=query["lexical_query"],
        vector=query_vector,
        target_vector="vector",
        alpha=alpha,
        limit=hybrid_top_k,
    )

    return [{
        "uuid": str(obj.uuid),
        "title": obj.properties["title"],
        "text": obj.properties["text"]
    } for obj in response.objects]


def merge_and_rerank_candidates(
    q_top_candidates: list[tuple[dict[str, str], float]],
    rewritten_q_top_candidates: list[tuple[dict[str, str], float]],
    reranker: CrossEncoder,
    q: dict[str, str],
) -> list[tuple[dict[str, str], float]]:
    """
    Merge and rerank candidates from the original and rewritten queries.

    Combines the ranked candidate lists from both retrieval attempts,
    removes duplicate passages using their UUIDs, and reranks all unique
    passages against the original semantic query to produce a consistent
    relevance ranking.

    Args:
        q_top_candidates: Ranked (document, score) tuples from the
            original retrieval query.
        rewritten_q_top_candidates: Ranked (document, score) tuples
            from the rewritten retrieval query.
        reranker: CrossEncoder used to score query-passage relevance.
        q: Original retrieval query containing a 'semantic_query' field.

    Returns:
        A list of (document, score) tuples containing unique passages,
        sorted by decreasing relevance to the original semantic query.
    """
    consolidated_candidates = []
    seen_ids = set()

    for doc, _ in q_top_candidates + rewritten_q_top_candidates:
        if doc["uuid"] not in seen_ids:
            consolidated_candidates.append(doc)
            seen_ids.add(doc["uuid"])

    return rerank(
        reranker,
        q["semantic_query"],
        consolidated_candidates,
        len(consolidated_candidates)
    )


def retrieve_single(
        client: weaviate.WeaviateClient,
        embedding_model: SentenceTransformer,
        reranker: CrossEncoder,
        slm: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        query: dict[str, str | list],
        collection_name: str = COLLECTION_NAME,
        alpha: float = HYBRID_SEARCH_ALPHA,
        hybrid_top_k: int = HYBRID_SEARCH_TOP_K,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        final_top_k: int = FINAL_CONTEXT_TOP_K,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """
        Retrieve context using the single-query strategy.

        Performs hybrid retrieval and reranks candidates against the original
        question. If the highest-ranked passage fails the relevance threshold,
        rewrites the retrieval query once, merges both candidate sets, and
        reranks the deduplicated results.

        Tracks search, embedding, reranking, and rewriting operations.

        Args:
            client: Connected Weaviate client.
            embedding_model: Model used to embed semantic queries.
            reranker: CrossEncoder used for passage reranking.
            slm: Small language model used to rewrite unsuccessful queries.
            tokenizer: Tokenizer associated with the SLM.
            query: Structured query containing the original question and one
                retrieval query.
            collection_name: Name of the Weaviate collection.
            alpha: Weight controlling the hybrid search balance.
            hybrid_top_k: Maximum number of passages retrieved by hybrid search.
            relevance_threshold: Minimum reranker score required for guaranteed
                passages before triggering a query rewrite.
            final_top_k: Maximum number of passages returned as context.

        Returns:
            Tuple containing the highest-ranked passages and a dictionary of retrieval operation counts.
    """
    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive")

    cost_stats = {
        "search_calls": 0,
        "embedding_calls": 0,
        "reranker_calls": 0,
        "reranked_pairs": 0,
        "rewrite_calls": 0
    }

    q = query["queries"][0]

    candidate_docs = hybrid_search(client, embedding_model, q, collection_name, alpha, hybrid_top_k)
    cost_stats["search_calls"] += 1
    cost_stats["embedding_calls"] += 1

    top_candidates = rerank(reranker, query["question"], candidate_docs, len(candidate_docs))
    if candidate_docs:
        cost_stats["reranker_calls"] += 1
        cost_stats["reranked_pairs"] += len(candidate_docs)

    # Retry once if retrieval relevance is insufficient
    if len(top_candidates) == 0 or top_candidates[0][1] < relevance_threshold:
        rewritten_q = rewrite_query(slm, tokenizer, q)
        cost_stats["rewrite_calls"] += 1

        rewritten_docs = hybrid_search(client, embedding_model, rewritten_q, collection_name, alpha, hybrid_top_k)
        cost_stats["search_calls"] += 1
        cost_stats["embedding_calls"] += 1

        rewritten_top_candidates = rerank(reranker, rewritten_q["semantic_query"], rewritten_docs, len(rewritten_docs))
        if rewritten_docs:
            cost_stats["reranker_calls"] += 1
            cost_stats["reranked_pairs"] += len(rewritten_docs)

        top_candidates = merge_and_rerank_candidates(top_candidates, rewritten_top_candidates, reranker, {"semantic_query": query["question"]})
        if top_candidates:
            cost_stats["reranker_calls"] += 1
            cost_stats["reranked_pairs"] += len(top_candidates)

    return [doc for doc, _ in top_candidates[:final_top_k]], cost_stats


def retrieve_decomposed(
        client: weaviate.WeaviateClient,
        embedding_model: SentenceTransformer,
        reranker: CrossEncoder,
        slm: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        query: dict[str, str | list],
        collection_name: str = COLLECTION_NAME,
        alpha: float = HYBRID_SEARCH_ALPHA,
        hybrid_top_k: int = HYBRID_SEARCH_TOP_K,
        subquery_top_k: int = SUBQUERY_TOP_K,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        final_context_top_k: int = FINAL_CONTEXT_TOP_K
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """
        Retrieve context using decomposed multi-query retrieval and
        tracks the number of hybrid searches, query embeddings, reranker calls,
        scored query-document pairs, and query rewrites.


        Each subquery is independently retrieved and locally reranked. If the
        top passages do not meet the relevance requirement, the
        subquery is rewritten and searched once more. A fixed number of passages
        from each subquery are guaranteed in the final context, while remaining
        candidates are deduplicated and globally reranked against the original
        question to fill the remaining context slots.

        Args:
            client: Connected Weaviate client.
            embedding_model: Model used to embed semantic queries.
            reranker: CrossEncoder used for passage reranking.
            slm: Small language model used to rewrite unsuccessful queries.
            tokenizer: Tokenizer associated with the SLM.
            query: Structured query containing the original question and
                decomposed retrieval queries.
            collection_name: Name of the Weaviate collection.
            alpha: Weight controlling the hybrid search balance.
            hybrid_top_k: Maximum number of candidates retrieved per search.
            subquery_top_k: Number of top-ranked passages selected from each
                subquery before deduplication. Shared passages occupy only
                one final context slot.
            relevance_threshold: Minimum reranker score required for guaranteed
                passages before triggering a query rewrite.
            final_context_top_k: Maximum number of passages in the final context.

        Returns:
            A deduplicated list of passages selected for the final RAG context and a dictionary of retrieval operation counts.
    """
    if subquery_top_k <= 0 or final_context_top_k <= 0:
        raise ValueError("Top-k values must be positive")

    if len(query['queries']) * subquery_top_k > final_context_top_k:
        raise ValueError("Maximum subquery selection exceeds the final context budget")

    cost_stats = {
        "search_calls": 0,
        "embedding_calls": 0,
        "reranker_calls": 0,
        "reranked_pairs": 0,
        "rewrite_calls": 0
    }

    relevant_docs = []
    relevant_doc_ids = set()
    global_candidates = []

    for q in query['queries']:
        candidate_docs = hybrid_search(client, embedding_model, q, collection_name, alpha, hybrid_top_k)
        cost_stats["search_calls"] += 1
        cost_stats["embedding_calls"] += 1

        top_candidates = rerank(reranker, q['semantic_query'], candidate_docs, len(candidate_docs))
        if candidate_docs:
            cost_stats["reranker_calls"] += 1
            cost_stats["reranked_pairs"] += len(candidate_docs)

        if not top_candidates or top_candidates[0][1] < relevance_threshold:
            rewritten_q = rewrite_query(slm, tokenizer, q) # rewrite query
            cost_stats["rewrite_calls"] += 1

            # find candidate docs related to rewritten_q
            rewritten_q_candidate_docs = hybrid_search(client, embedding_model, rewritten_q, collection_name, alpha, hybrid_top_k)
            cost_stats["search_calls"] += 1
            cost_stats["embedding_calls"] += 1

            # rerank and return top candidates against rewritten_q
            rewritten_q_top_candidates = rerank(reranker, rewritten_q['semantic_query'], rewritten_q_candidate_docs, len(rewritten_q_candidate_docs))
            if rewritten_q_candidate_docs:
                cost_stats["reranker_calls"] += 1
                cost_stats["reranked_pairs"] += len(rewritten_q_candidate_docs)

            # combine top candidates from rewritten_q and q; deduplicate; and rerank
            top_candidates = merge_and_rerank_candidates(top_candidates, rewritten_q_top_candidates, reranker, q)
            if top_candidates:
                cost_stats["reranker_calls"] += 1
                cost_stats["reranked_pairs"] += len(top_candidates)

        # Merge top candidates from multiple subqueries into a single relevant_docs list without duplication.
        # This ensures that all subqueries get coverage in relevant_docs list
        for doc, _ in top_candidates[:subquery_top_k]:
            if doc['uuid'] not in relevant_doc_ids:
                relevant_docs.append(doc)
                relevant_doc_ids.add(doc['uuid'])
        global_candidates.extend(top_candidates[subquery_top_k:])

    # deduplicating the global_candidates list. This list contains docs returned by hybrid search
    # which are ranked low in reranking against sub-queires and so not included in relevant_docs list above.
    seen_ids = set()
    deduped_global_candidates = []
    for doc, _ in global_candidates:
        if doc['uuid'] not in relevant_doc_ids and doc['uuid'] not in seen_ids:
            seen_ids.add(doc['uuid'])
            deduped_global_candidates.append(doc)

    # filling available slots in relevant_docs list with the docs from global_candidates list which
    # are most relevant to the original question.
    available_global_spots = final_context_top_k - len(relevant_docs)
    if available_global_spots > 0 and deduped_global_candidates:
        global_top_candidates = rerank(reranker, query['question'], deduped_global_candidates, available_global_spots)
        cost_stats["reranker_calls"] += 1
        cost_stats["reranked_pairs"] += len(deduped_global_candidates)

        relevant_docs = relevant_docs + [c[0] for c in global_top_candidates]

    # final reranking to rank the most relevant documents highly
    relevant_docs_ranked = rerank(reranker, query['question'], relevant_docs, final_context_top_k)
    if relevant_docs:
        cost_stats["reranker_calls"] += 1
        cost_stats["reranked_pairs"] += len(relevant_docs)

    return [doc for doc, _ in relevant_docs_ranked], cost_stats
