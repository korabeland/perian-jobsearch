"""Semantic similarity via nomic-embed-text embeddings.

Provides embedding-based matching for eval scoring (L1 requirements, L6 fact
coverage, L6 hallucination detection). Uses Ollama /api/embed for batch
embedding. Caches reference embeddings within a single eval run.
"""

import math

import requests

_EMBED_MODEL = "nomic-embed-text"
_OLLAMA_BASE = "http://localhost:11434"

# Module-level cache: text -> embedding vector.  Reference texts (expected
# facts, requirements) don't change across models, so we cache them for the
# duration of the process.
_embedding_cache: dict[str, list[float]] = {}


def embed_texts(texts: list[str], base_url: str = _OLLAMA_BASE) -> list[list[float]]:
    """Batch-embed texts via Ollama /api/embed.

    Returns a list of 768-dim vectors, one per input text.
    Uses module-level cache for previously-seen texts.
    """
    # Split into cached and uncached
    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(texts):
        if text in _embedding_cache:
            results[i] = _embedding_cache[text]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        resp = requests.post(
            f"{base_url}/api/embed",
            json={"model": _EMBED_MODEL, "input": uncached_texts},
            timeout=60,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]

        for idx, text, emb in zip(uncached_indices, uncached_texts, embeddings):
            _embedding_cache[text] = emb
            results[idx] = emb

    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_match(
    candidate: str, reference: str, threshold: float = 0.80
) -> tuple[bool, float]:
    """Check if candidate semantically matches reference.

    Returns (matches, similarity_score).
    """
    embeddings = embed_texts([candidate, reference])
    score = cosine_similarity(embeddings[0], embeddings[1])
    return score >= threshold, round(score, 4)


def best_semantic_match(
    candidate: str,
    references: list[str],
    threshold: float = 0.80,
) -> tuple[bool, float, str | None]:
    """Find the best semantic match for candidate among references.

    Returns (matches, best_score, best_reference_or_None).
    """
    if not references:
        return False, 0.0, None

    all_texts = [candidate] + references
    embeddings = embed_texts(all_texts)
    candidate_emb = embeddings[0]
    ref_embeddings = embeddings[1:]

    best_score = -1.0
    best_ref = None
    for ref_text, ref_emb in zip(references, ref_embeddings):
        score = cosine_similarity(candidate_emb, ref_emb)
        if score > best_score:
            best_score = score
            best_ref = ref_text

    return best_score >= threshold, round(best_score, 4), best_ref
