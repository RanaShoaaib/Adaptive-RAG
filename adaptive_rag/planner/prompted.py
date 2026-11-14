import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from adaptive_rag.config import SLM_MAX_NEW_TOKENS, SUBQUERY_TOP_K, FINAL_CONTEXT_TOP_K
from adaptive_rag.models.lm import generate


def validate_retrieval_query(query: dict) -> None:
    """Validate a lexical/semantic retrieval query."""
    if not isinstance(query, dict):
        raise ValueError("Each retrieval query must be a JSON object.")

    for field in ("lexical_query", "semantic_query"):
        value = query.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")


def validate_query_plan(result: dict) -> None:
    """Validate the SLM-generated ONCE/SPLIT retrieval plan."""
    if not isinstance(result, dict):
        raise ValueError("Retrieval plan must be a JSON object.")

    strategy = result.get("strategy")
    queries = result.get("queries")

    if strategy not in {"once", "split"}:
        raise ValueError("Strategy must be 'once' or 'split'.")

    if not isinstance(queries, list):
        raise ValueError("Queries must be a list.")

    if strategy == "once" and len(queries) != 1:
        raise ValueError("ONCE must contain exactly one retrieval query.")

    if strategy == "split" and len(queries) != 2:
        raise ValueError("SPLIT must contain exactly two retrieval queries.")

    if len(queries) * SUBQUERY_TOP_K > FINAL_CONTEXT_TOP_K:
        raise ValueError(
            "Subquery selection exceeds the final context budget."
        )

    for query in queries:
        validate_retrieval_query(query)


def generate_valid_json(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    validator,
    max_retries: int = 1,
) -> dict:
    """
    Generate JSON and validate its schema.

    If generation fails JSON parsing or schema validation,
    retry with a corrective instruction.

    Raises ValueError if all attempts fail.
    """

    for attempt in range(max_retries + 1):
        response = generate(model, tokenizer, messages, max_new_tokens=SLM_MAX_NEW_TOKENS)

        try:
            cleaned_response = response.strip()

            if cleaned_response.startswith("```"):
                lines = cleaned_response.splitlines()
                if lines[-1].strip() == "```":
                    cleaned_response = "\n".join(lines[1:-1]).strip()

            result = json.loads(cleaned_response)
            validator(result)
            return result

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            if attempt == max_retries:
                raise ValueError(
                    f"Invalid SLM output after {attempt + 1} attempts. "
                    f"Last error: {exc}. "
                    f"Last response: {response!r}"
                ) from exc

            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {exc}. "
                        "Start over and generate a new JSON object. "
                        "For SPLIT, return EXACTLY TWO retrieval queries. "
                        "For ONCE, return EXACTLY ONE retrieval query. "
                        "Keep queries concise and relevant to the original question. "
                        "Return only complete JSON without markdown or explanations."
                    ),
                },
            ]

    raise RuntimeError("Unexpected JSON generation failure.")


def generate_query(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
) -> dict[str, str | list]:
    """
    Generate and validate an ONCE/SPLIT retrieval plan.

    Returns a dictionary containing the original question,
    strategy, and lexical/semantic retrieval queries.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval planner for a RAG system. "
                "Decide whether a question requires one retrieval query "
                "or should be decomposed into multiple retrieval queries. "
                "Return only valid JSON with no explanation or additional text."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Write retrieval queries for a RAG system based on the question provided.
                
                Return exactly this JSON structure:
                {{
                    "strategy": "once" or "split",
                    "queries": [
                        {{
                            "lexical_query": "...",
                            "semantic_query": "..."
                        }}
                    ]
                }}
                
                Strategy:
                - Use "once" when a single retrieval query can reasonably retrieve the evidence needed to answer the question.
                - Use "split" when separate retrievals for different subjects, entities, or intermediate facts would provide complementary evidence that, when combined, would better support answering the original question.
                - For comparisons involving two subjects, consider retrieving the relevant fact about each subject separately.
                - Do not split when the decomposed queries would be synonymous in meaning.
                
                Queries:
                - Always return a list of query objects.
                - For "once", return exactly one query.
                - For "split", return exactly two queries.
                - Each query must represent a distinct information need.
                - Do not generate duplicate or equivalent queries.
                - Do not change what the question asks or introduce unsupported assumptions.
                - Both query fields must contain non-empty strings.
                - lexical_query should preserve exact entity names and relevant nouns and keywords from the original question for BM25 retrieval.
                Collectively, the lexical queries should cover all essential entities and keywords from the original question.
                - semantic_query should express the information need naturally for semantic vector search.
                  
                
                The following examples illustrate the expected query generation behavior 
                and JSON output format. Use them as guidance, not as queries to reproduce.
                
                Examples:
                Example 1 — "once" (Single-hop question)
                Question: Who wrote the novel Pride and Prejudice?
                Output:
                {{
                    "strategy": "once",
                    "queries": [
                        {{
                            "lexical_query": "Pride and Prejudice author",
                            "semantic_query": "Who wrote the novel Pride and Prejudice?"
                        }}
                    ]
                }}
                
                Example 2 — "split"
                Question: Which film was released earlier, Titanic or Avatar?
                Output:
                {{
                    "strategy": "split",
                    "queries": [
                        {{
                            "lexical_query": "Titanic film release date",
                            "semantic_query": "When was the film Titanic released?"
                        }},
                        {{
                            "lexical_query": "Avatar film release date",
                            "semantic_query": "When was the film Avatar released?"
                        }}
                    ]
                }}
                
                Now generate the retrieval plan for the following question. Return only the JSON object.
                Question: {question}
                """,
        },
    ]

    result = generate_valid_json(model, tokenizer, messages, validate_query_plan)
    result["question"] = question
    return result


def rewrite_query(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: dict[str, str],
) -> dict[str, str]:
    """
    Rewrite an unsuccessful retrieval query.

    Returns validated lexical and semantic queries.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite queries meant for RAG to improve search. "
                "Return only valid JSON with no explanation or additional text."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Rewrite the following failed retrieval queries into improved search queries.
                
                Current semantic query:
                {query["semantic_query"]}
                
                Current lexical query:
                {query["lexical_query"]}
                
                semantic_query:
                A natural-language query that preserves the meaning and context
                and is optimized for semantic vector search.
                
                lexical_query:
                A concise keyword-based query containing important entities
                and terms, optimized for BM25 lexical search.
                
                Return exactly this JSON structure:
                {{
                    "semantic_query": "...",
                    "lexical_query": "..."
                }}
                """,
        },
    ]

    return generate_valid_json(model, tokenizer, messages, validate_retrieval_query)


def generate_always_once_query(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
) -> dict[str, str | list]:
    """Generate exactly one lexical/semantic retrieval query, regardless of complexity."""
    messages = [
        {
            "role": "system",
            "content": (
                "You generate a single retrieval query for a RAG system. "
                "Always use the `once` strategy, even for multi-hop questions. "
                "Return only valid JSON with no additional text."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Create ONE retrieval query for the question below.
                The query should capture the full information need as effectively
                as possible without decomposing it into subqueries.
                              
                Return exactly this JSON structure:
                {{
                    "strategy": "once",
                    "queries": [
                        {{
                            "lexical_query": "...",
                            "semantic_query": "..."
                        }}
                    ]
                }}

                - lexical_query: important entities and keywords from the question for BM25.
                - semantic_query: natural-language information need derived from the question for vector search.
                - Both fields must be non-empty strings.
                - Do not return additional fields or explanations.

                The following examples illustrate the expected query generation behavior 
                and JSON output format. Use them as guidance, not as queries to reproduce.
                
                Examples:
                Example 1 — Simple question
                Question: Who painted the Mona Lisa?
                Output:
                {{
                    "strategy": "once",
                    "queries": [
                        {{
                            "lexical_query": "Mona Lisa painter",
                            "semantic_query": "Who painted the Mona Lisa?"
                        }}
                    ]
                }}
                
                Example 2 — Multi-hop question
                Question: In which country was the author of The Hobbit born?
                Output:
                {{
                    "strategy": "once",
                    "queries": [
                        {{
                            "lexical_query": "The Hobbit author birthplace country",
                            "semantic_query": "In which country was the author of The Hobbit born?"
                        }}
                    ]
                }}
                
                Now generate the retrieval plan for the following question. Return only the JSON object.
                Question: {question}
            """,
        },
    ]

    def validate_once(plan: dict) -> None:
        validate_query_plan(plan)
        if plan["strategy"] != "once":
            raise ValueError("Always-ONCE planner must return strategy='once'.")

    result = generate_valid_json(model, tokenizer, messages, validate_once)
    result["question"] = question
    return result


def generate_always_split_query(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
) -> dict[str, str | list]:
    """Force decomposition into two or more distinct retrieval queries."""
    max_subqueries = FINAL_CONTEXT_TOP_K // SUBQUERY_TOP_K
    if max_subqueries < 2:
        raise ValueError("Current context budget cannot accommodate a SPLIT plan.")

    messages = [
        {
            "role": "system",
            "content": (
                "You decompose questions into retrieval queries for a RAG system. "
                "Always use the `split` strategy, even for simple questions. "
                "Return only valid JSON with no additional text."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Decompose the question into exactly TWO distinct retrieval information needs. 
                For simple questions, split into complementary
                factual aspects rather than merely paraphrasing the same query.
                Do not answer the question.

                Return exactly this JSON structure:
                {{
                    "strategy": "split",
                    "queries": [
                        {{
                            "lexical_query": "...",
                            "semantic_query": "..."
                        }},
                        {{
                            "lexical_query": "...",
                            "semantic_query": "..."
                        }}
                    ]
                }}

                - Generate and return exactly 2 query objects.
                - Keep each query concise.
                - For multi-hop questions, decompose into distinct information needs.
                - For single-hop questions, generate complementary retrieval
                  perspectives while preserving the original information need.
                - Avoid duplicate or equivalent queries whenever possible.
                - lexical_query: important entities and keywords from the question for use in BM25 retrieval.
                - semantic_query: natural-language information need based on the question for vector search.
                - Both fields must be non-empty strings.
                - Do not return additional fields or explanations.
                
                The following examples illustrate the expected query generation behavior 
                and JSON output format. Use them as guidance, not as queries to reproduce.
                Examples:
                Example 1 — Comparison question
                Question: Which film was released earlier, Titanic or Avatar?
                Output:
                {{
                    "strategy": "split",
                    "queries": [
                        {{
                            "lexical_query": "Titanic film release date",
                            "semantic_query": "When was the film Titanic released?"
                        }},
                        {{
                            "lexical_query": "Avatar film release date",
                            "semantic_query": "When was the film Avatar released?"
                        }}
                    ]
                }}
                
                Example 2 — Bridge question
                Question: In which country was the author of The Hobbit born?
                Output:
                {{
                    "strategy": "split",
                    "queries": [
                        {{
                            "lexical_query": "The Hobbit author",
                            "semantic_query": "Who wrote The Hobbit?"
                        }},
                        {{
                            "lexical_query": "The Hobbit author birthplace country",
                            "semantic_query": "In which country was the author of The Hobbit born?"
                        }}
                    ]
                }}
                
                Now generate the retrieval plan for the following question. Return only the JSON object.
                Question: {question}
            """,
        },
    ]

    def validate_split(plan: dict) -> None:
        validate_query_plan(plan)
        if plan["strategy"] != "split":
            raise ValueError("Always-SPLIT planner must return strategy='split'.")

    result = generate_valid_json(model, tokenizer, messages, validate_split)
    result["question"] = question
    return result
