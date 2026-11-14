import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize an answer for standard QA evaluation."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punctuation(text):
        return "".join(
            char for char in text
            if char not in string.punctuation
        )

    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """Calculate normalized exact match."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Calculate token-level F1 between prediction and reference."""

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def evaluate_answer(prediction: str, references: list[str]) -> dict[str, float]:
    """
    Evaluate an answer against multiple acceptable references.

    Returns the maximum EM and F1 across references.
    """

    if not references:
        raise ValueError("references must not be empty")

    return {
        "exact_match": max(
            exact_match(prediction, ref)
            for ref in references
        ),
        "f1": max(
            token_f1(prediction, ref)
            for ref in references
        )
    }
