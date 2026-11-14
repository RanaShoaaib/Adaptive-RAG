from adaptive_rag.models.lm import generate
from adaptive_rag.config import ANSWERING_MAX_NEW_TOKENS
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_augmented_prompt(question: str, docs: list[dict]) -> list[dict[str, str]]:
    """
        Construct a context-augmented chat prompt for answer generation.

        Args:
            question: Original question to be answered.
            docs: Retrieved passages containing title and text fields.

        Returns:
            A list of chat messages containing the answering instructions,
            original question, and formatted retrieved context.
    """
    context = "\n\n".join(
        f"[{i}] Title: {doc['title']}\n{doc['text']}"
        for i, doc in enumerate(docs, start=1)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the provided context. "
                "Identify the exact phrase in the context that answers the question. "
                "Return only that phrase, without explanation. "
                "If the answer cannot be determined, return 'Unknown'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Context:\n{context}"
            )
        }
    ]
    return messages


def generate_answer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str, docs: list[dict]) -> str:
    """
        Generate an answer to a question using retrieved context.

        Constructs an augmented prompt and passes it to the answering
        language model for generation.

        Args:
            model: Loaded causal language model used for answering.
            tokenizer: Tokenizer associated with the answering model.
            question: Original question to be answered.
            docs: Retrieved passages containing title and text fields.

        Returns:
            The generated answer as a string.
    """
    prompt = generate_augmented_prompt(question, docs)
    response = generate(model, tokenizer, prompt, max_new_tokens=ANSWERING_MAX_NEW_TOKENS)
    return response.strip()
