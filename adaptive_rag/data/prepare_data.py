import uuid
import numpy as np
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from adaptive_rag.config import (
    HOTPOTQA_DATASET_PATH,
    HOTPOTQA_DATASET_NAME,
    SQUAD_DATASET_PATH,
    VAL_SIZE,
    SEED,
    TRAIN_SUBSET_PATH,
    VAL_SUBSET_PATH,
    TEST_SUBSET_PATH,
)


SUBSET_PATHS = {
    "train": TRAIN_SUBSET_PATH,
    "val": VAL_SUBSET_PATH,
    "test": TEST_SUBSET_PATH,
}


def chunk_from_sentence_list(sentence_list: list[str]) -> str:
    """
        Combine a list of sentences into a single text chunk.
        Leading and trailing whitespace is removed from each sentence before
        the sentences are joined with spaces.

        Args:
            sentence_list: Sentences belonging to a single context passage.

        Returns:
            A single string containing the joined sentences.
    """
    return " ".join(sentence.strip() for sentence in sentence_list)


def make_passage_id(title: str, text: str) -> str:
    """
        Generate a deterministic UUID for a passage.
        Passages with identical titles and text receive the same UUID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{title}\n{text}"))


def make_hotpotqa_chunks(titles: list[str], sentences: list[list[str]]) -> list[dict[str, str]]:
    """
        Convert context titles and sentence lists into passage-level chunks.
        Each title is paired with its corresponding list of sentences. The
        sentences are combined into a single text passage.

        Args:
            titles: Titles of the context passages.
            sentences: Lists of sentences corresponding to each title.

        Returns:
            A list of passage dictionaries containing ``passage_id``, ``title``,
            and ``text`` fields.
    """
    if len(titles) != len(sentences):
        raise ValueError("The number of titles and sentences must match")

    chunks = []
    for title, sentence in zip(titles, sentences):
        text = chunk_from_sentence_list(sentence)
        chunks.append({
            "passage_id": make_passage_id(title, text),
            "title": title,
            "text": text,
        })
    return chunks


def build_hotpotqa_corpus(dataset: Dataset) -> tuple[Dataset, list[dict[str, str]]]:
    """
        Build a HotpotQA passage corpus and annotate questions with gold passage IDs.

        Each context entry is converted into a passage with a deterministic UUID.
        Passages whose titles occur in the question's supporting facts are marked
        as gold evidence.

        Args:
            dataset: HotpotQA dataset containing context and supporting_facts.

        Returns:
            A tuple containing:
                - Dataset with an added gold_passage_ids column.
                - List of passage dictionaries for corpus ingestion.
    """
    required_columns = {"context", "supporting_facts"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError(f"HotpotQA dataset must contain {required_columns}")

    chunks = []
    gold_passage_ids = []

    for row in dataset:
        context = row["context"]
        passages = make_hotpotqa_chunks(
            context["title"],
            context["sentences"]
        )
        chunks.extend(passages)

        supporting_titles = set(row["supporting_facts"]["title"])
        context_titles = set(context["title"])
        missing_titles = supporting_titles - context_titles
        if missing_titles:
            raise ValueError(f"Supporting titles missing from context: {missing_titles}")

        gold_ids = list(dict.fromkeys(
            passage["passage_id"]
            for passage in passages
            if passage["title"] in supporting_titles
        ))
        gold_passage_ids.append(gold_ids)

    # Attach the precomputed IDs using dataset row indices.
    def add_gold_ids(batch, indices):
        batch["gold_passage_ids"] = [gold_passage_ids[i] for i in indices]
        return batch

    updated_dataset = dataset.map(
        add_gold_ids,
        batched=True,
        with_indices=True
    )

    return updated_dataset, chunks


def build_squad_corpus(dataset: Dataset) -> tuple[Dataset, list[dict[str, str]]]:
    """
        Build a SQuAD passage corpus and annotate questions with gold passage IDs.

        Each question's context is treated as its gold passage. A deterministic
        UUID is generated from the context title and text.

        Args:
            dataset: SQuAD dataset containing title and context.

        Returns:
            A tuple containing:
                - Dataset with an added gold_passage_ids column.
                - List of passage dictionaries for corpus ingestion.
    """
    required_columns = {"title", "context"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError(f"SQuAD dataset must contain {required_columns}")

    def add_gold_ids(batch):
        batch["gold_passage_ids"] = [
            [make_passage_id(title, text)]
            for title, text in zip(
                batch["title"],
                batch["context"]
            )
        ]
        return batch

    updated_dataset = dataset.map(add_gold_ids,batched=True)

    chunks = []

    for row in updated_dataset:
        chunks.append({
            "passage_id": row["gold_passage_ids"][0],
            "title": row["title"],
            "text": row["context"]
        })

    return updated_dataset, chunks


def combine_and_deduplicate_corpus(corpus_list: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """
    Combine corpora and remove passages with duplicate passage IDs.
    """
    unique_passages = {}

    for corpus in corpus_list:
        for passage in corpus:
            passage_id = passage["passage_id"]

            if passage_id not in unique_passages:
                unique_passages[passage_id] = passage

    return list(unique_passages.values())


def normalize_data(batch: dict, dataset_type: str) -> dict:
    """
        Normalize a batch of HotpotQA or SQuAD questions to a common schema.

        HotpotQA's single answer string is converted into a one-element list.
        SQuAD's reference answer strings are retained as a list.

        Args:
            batch: Batch of question records.
            dataset_type: Either "hotpotqa" or "squad".

        Returns:
            Dictionary containing id, source, question, answers,
            and gold_passage_ids.
    """
    dataset_type = dataset_type.lower().strip()

    if dataset_type not in {"hotpotqa", "squad"}:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if dataset_type == "hotpotqa":
        answers = [[answer] for answer in batch["answer"]]
    else:
        answers = [item["text"] for item in batch["answers"]]

    return {
        "id": batch["id"],
        "source": [dataset_type] * len(batch["id"]),
        "question": batch["question"],
        "answers": answers,
        "gold_passage_ids": batch["gold_passage_ids"],
    }


def normalize_and_concatenate_datasets(
    datasets: dict[str, dict[str, Dataset]],
    val_size: float,
    seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    """
        Normalize and combine HotpotQA and SQuAD question datasets.

        Each source's training set is independently divided into training and
        validation partitions. The original validation split is reserved for
        testing. Corresponding partitions are concatenated and shuffled.

        Args:
            datasets: Nested dictionary of source datasets under train and test.
            val_size: Fraction of each training dataset reserved for validation.
            seed: Random seed for reproducible splitting and shuffling.

        Returns:
            Combined training, validation, and test datasets.
    """

    train_parts = []
    val_parts = []
    test_parts = []

    for source in ("hotpotqa", "squad"):

        train_ds = datasets["train"][source]
        test_ds = datasets["test"][source]

        # Normalize training questions.
        train_ds = train_ds.map(
            normalize_data,
            batched=True,
            fn_kwargs={"dataset_type": source},
            remove_columns=train_ds.column_names
        )

        # Normalize held-out test questions.
        test_ds = test_ds.map(
            normalize_data,
            batched=True,
            fn_kwargs={"dataset_type": source},
            remove_columns=test_ds.column_names
        )

        # Split training questions into train and validation.
        split = train_ds.train_test_split(
            test_size=val_size,
            seed=seed
        )

        train_parts.append(split["train"])
        val_parts.append(split["test"])
        test_parts.append(test_ds)

    # Combine the corresponding splits from both datasets.
    ds_train = concatenate_datasets(train_parts).shuffle(seed=seed)
    ds_val = concatenate_datasets(val_parts).shuffle(seed=seed)
    ds_test = concatenate_datasets(test_parts).shuffle(seed=seed)

    return ds_train, ds_val, ds_test



def get_data_and_corpus(
        *,
        hotpot_dataset_path: str = HOTPOTQA_DATASET_PATH,
        hotpot_name: str = HOTPOTQA_DATASET_NAME,
        squad_dataset_path: str = SQUAD_DATASET_PATH,
        val_size: float = VAL_SIZE, seed: int = SEED
) -> tuple[Dataset, Dataset, Dataset, list[dict[str, str]]]:
    """
        Load, prepare, and combine HotpotQA and SQuAD datasets.

        Builds a shared deduplicated passage corpus, assigns deterministic
        passage IDs, annotates gold evidence, and produces normalized
        training, validation, and test question datasets.

        The original validation splits are used as held-out test sets.

        Returns:
            A tuple containing:
                - Combined training questions.
                - Combined validation questions.
                - Combined held-out test questions.
                - Shared deduplicated passage corpus.
    """
    hotpotqa_ds = load_dataset(hotpot_dataset_path, hotpot_name)
    hotpotqa_ds_train, hotpotqa_ds_test = hotpotqa_ds['train'], hotpotqa_ds['validation']
    hotpotqa_ds_train, hotpotqa_train_corpus = build_hotpotqa_corpus(hotpotqa_ds_train)
    hotpotqa_ds_test, hotpotqa_test_corpus = build_hotpotqa_corpus(hotpotqa_ds_test)

    squad_ds = load_dataset(squad_dataset_path)
    squad_ds_train, squad_ds_test = squad_ds['train'], squad_ds['validation']
    squad_ds_train, squad_train_corpus = build_squad_corpus(squad_ds_train)
    squad_ds_test, squad_test_corpus = build_squad_corpus(squad_ds_test)

    corpus = combine_and_deduplicate_corpus(
        [
            hotpotqa_train_corpus,
            hotpotqa_test_corpus,
            squad_train_corpus,
            squad_test_corpus,
        ]
    )

    datasets = {
        'train': {'hotpotqa': hotpotqa_ds_train, 'squad': squad_ds_train},
        'test': {'hotpotqa': hotpotqa_ds_test, 'squad': squad_ds_test},
    }

    ds_train, ds_val, ds_test = normalize_and_concatenate_datasets(
        datasets=datasets,
        val_size=val_size,
        seed=seed
    )

    return ds_train, ds_val, ds_test, corpus


def sample_dataset(dataset: Dataset, n: int, seed: int = 42, strategy: str = "stratified") -> Dataset:
    """
    Samples the passed Hugging Face Dataset based on its 'source' column.
    Returns Dataset with `n` samples.

    Strategies:
        stratified: Preserve the original source proportions.
        balanced:   Allocate samples equally across sources.
    """
    if strategy not in {"stratified", "balanced"}:
        raise ValueError("strategy must be 'stratified' or 'balanced'.")

    if not 0 < n <= len(dataset):
        raise ValueError("n must be between 1 and the dataset size.")

    if "source" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'source' column.")

    rng = np.random.default_rng(seed)

    sources, inverse, sizes = np.unique(dataset["source"], return_inverse=True, return_counts=True)

    if strategy == "stratified":
        weights = sizes / sizes.sum()
    else:
        weights = np.full(len(sources), 1 / len(sources))

    # Largest-remainder allocation ensures counts sum exactly to n.
    raw_counts = n * weights
    counts = np.floor(raw_counts).astype(int)

    remainder = n - counts.sum()
    order = np.argsort(-(raw_counts - counts), kind="stable")
    counts[order[:remainder]] += 1

    if np.any(counts > sizes):
        raise ValueError("Insufficient examples for the requested allocation.")

    selected_indices = []

    for i, count in enumerate(counts):
        source_indices = np.flatnonzero(inverse == i)
        selected_indices.extend(rng.choice(source_indices, size=count, replace=False).tolist())

    rng.shuffle(selected_indices)

    return dataset.select(selected_indices)


def save_subset(dataset: Dataset, path: str | Path) -> None:
    """Save a Hugging Face Dataset subset to a Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))


def load_subset(path: str | Path) -> Dataset:
    """Load a saved Parquet subset as a Hugging Face Dataset."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset subset not found: {path}")

    return Dataset.from_parquet(str(path))


def prepare_subsets(subset_configs: dict[str, dict], seed: int = SEED) -> None:
    """Create only the requested fixed subsets from full source datasets.

    Example: {"train": {"n": 100, "strategy": "balanced"},
              "val": {"n": 20, "strategy": "balanced"}}
    """
    if not subset_configs:
        raise ValueError("subset_configs cannot be empty")
    unknown = set(subset_configs) - set(SUBSET_PATHS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")

    ds_train, ds_val, ds_test, _ = get_data_and_corpus()
    datasets = {"train": ds_train, "val": ds_val, "test": ds_test}

    for split, settings in subset_configs.items():
        subset = sample_dataset(
            datasets[split],
            n=settings["n"],
            seed=seed,
            strategy=settings["strategy"],
        )
        save_subset(subset, SUBSET_PATHS[split])
        print(f"Saved {len(subset)} {split} questions to {SUBSET_PATHS[split]}")