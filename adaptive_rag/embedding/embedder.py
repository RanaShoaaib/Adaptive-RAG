from sentence_transformers import SentenceTransformer
from adaptive_rag.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, QUERY_PREFIX, PASSAGE_PREFIX

def get_embedding_model(model: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
        Load and return the configured sentence-transformer embedding model.

        Args:
            model: Name of model used to make the SentenceTransformer.
        Returns:
            The SentenceTransformer model specified by model.
    """
    return SentenceTransformer(model)


def embed_queries(model: SentenceTransformer, queries: list[str]) -> list[list[float]]:
    """
        Generate normalized embeddings for semantic queries.

        Each query is prepended with ``QUERY_PREFIX`` before encoding. Queries are
        encoded in batches of ``EMBEDDING_BATCH_SIZE``.
        Pass only the semantic query text.

        Args:
            model: SentenceTransformer model used to encode the queries.
            queries: Semantic query texts, without the prefix.

        Returns:
            A list of normalized embedding vectors, one for each query, in
            the same order as ``queries``.
    """
    embeddings = model.encode(
        [f"{QUERY_PREFIX}{q}" for q in queries],
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_passages(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
        Generate normalized embeddings for corpus passages.

        Each text is prepended with ``PASSAGE_PREFIX`` before encoding. For
        ``bge-base-en-v1.5`` the prefix is empty, since the instruction applies
        to queries only. The prefix is applied here and nowhere else, so the
        ingestion and query paths cannot drift apart. Passages are encoded in
        batches of ``EMBEDDING_BATCH_SIZE``. Embeddings are L2-normalized, which
        matches the cosine distance metric set on the collection, and are
        returned as Python lists for storage in the vector database.

        Args:
            model: SentenceTransformer model used to encode the passages.
            texts: Passage texts to embed.

        Returns:
            A list of normalized embedding vectors, one for each passage.
    """
    embeddings = model.encode(
        [f"{PASSAGE_PREFIX}{t}" for t in texts],
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()
