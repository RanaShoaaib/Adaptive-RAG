"""Invariants on adaptive_rag.config that the retrieval/planner code relies on."""
from pathlib import Path

import adaptive_rag
from adaptive_rag import config as cfg


def test_hybrid_alpha_and_threshold_are_probabilities():
    assert 0.0 <= cfg.HYBRID_SEARCH_ALPHA <= 1.0
    assert 0.0 <= cfg.RELEVANCE_THRESHOLD <= 1.0


def test_context_budget_allows_a_two_way_split():
    """validate_query_plan/retrieve_decomposed need 2 * SUBQUERY_TOP_K <= FINAL_CONTEXT_TOP_K."""
    assert cfg.SUBQUERY_TOP_K > 0
    assert 2 * cfg.SUBQUERY_TOP_K <= cfg.FINAL_CONTEXT_TOP_K


def test_candidate_pool_is_not_smaller_than_final_context():
    assert cfg.HYBRID_SEARCH_TOP_K >= cfg.FINAL_CONTEXT_TOP_K


def test_positive_sizes():
    for value in (cfg.EMBEDDING_BATCH_SIZE, cfg.INGEST_CHUNK_SIZE, cfg.SLM_MAX_NEW_TOKENS,
                  cfg.ANSWERING_MAX_NEW_TOKENS, cfg.FINAL_CONTEXT_TOP_K, cfg.HYBRID_SEARCH_TOP_K):
        assert isinstance(value, int) and value > 0


def test_validation_fraction_and_seed():
    assert 0.0 < cfg.VAL_SIZE < 1.0
    assert isinstance(cfg.SEED, int)


def test_subset_paths_are_distinct_parquet_files_in_datasets_dir():
    paths = [cfg.TRAIN_SUBSET_PATH, cfg.VAL_SUBSET_PATH, cfg.TEST_SUBSET_PATH]
    assert len(set(paths)) == 3
    for path in paths:
        assert path.suffix == ".parquet"
        assert path.parent == cfg.DATASETS_DIR


def test_root_dir_is_the_repository_root():
    package_dir = Path(adaptive_rag.__file__).resolve().parent
    assert cfg.ROOT_DIR.resolve() == package_dir.parent
    assert cfg.DATASETS_DIR == cfg.ROOT_DIR / "subsets"


def test_instruction_prefix_is_query_only():
    assert cfg.QUERY_PREFIX.strip()
    assert cfg.PASSAGE_PREFIX == ""
