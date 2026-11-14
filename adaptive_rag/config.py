from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")


DATASETS_DIR = ROOT_DIR / "subsets"
TRAIN_SUBSET_PATH = DATASETS_DIR / "train_subset.parquet"
VAL_SUBSET_PATH = DATASETS_DIR / "val_subset.parquet"
TEST_SUBSET_PATH = DATASETS_DIR / "test_subset.parquet"

HOTPOTQA_DATASET_PATH = "hotpotqa/hotpot_qa"
HOTPOTQA_DATASET_NAME = "distractor"

SQUAD_DATASET_PATH = "rajpurkar/squad"

VAL_SIZE = 0.1
SEED = 42

COLLECTION_NAME = "AdaptiveRag"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 128

INGEST_CHUNK_SIZE = 1024

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = ""

HYBRID_SEARCH_TOP_K = 20
HYBRID_SEARCH_ALPHA = 0.7

RERANKING_MODEL = "BAAI/bge-reranker-base"

RELEVANCE_THRESHOLD = 0.8
SUBQUERY_TOP_K = 2
FINAL_CONTEXT_TOP_K = 7

SLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SLM_MAX_NEW_TOKENS = 256

ANSWERING_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ANSWERING_MAX_NEW_TOKENS = 128
