# =============================================================
# tfidf_cosine_scoring.py
#
# Stage: TF-IDF Cosine Scoring
# Computes a TF-IDF vector for every sentence, builds a
# pseudo-document for each LDA topic (concatenation of its
# top-K words weighted by probability), then measures cosine
# similarity between each sentence vector and each topic vector.
#
# Implemented entirely from scratch — no sklearn or similar.
# =============================================================

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from lda_topic_modelling import LDAResult


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vector    = Dict[str, float]                     # sparse TF-IDF vector
ScoreMatrix = List[List[float]]                  # [sentence][topic] → float


# ---------------------------------------------------------------------------
# TF-IDF implementation
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Simple alpha-only tokeniser, lowercase."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 2]


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency: count / total_terms (raw TF)."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total  = len(tokens)
    return {w: c / total for w, c in counts.items()}


def _compute_idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """
    Inverse document frequency (smooth):
        IDF(t) = log( (1 + N) / (1 + df(t)) ) + 1
    where N = number of documents and df(t) = document frequency.
    """
    N  = len(corpus_tokens)
    df : Dict[str, int] = {}
    for tokens in corpus_tokens:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1

    idf: Dict[str, float] = {}
    for word, freq in df.items():
        idf[word] = math.log((1 + N) / (1 + freq)) + 1.0

    return idf


def _tfidf_vector(tf: Dict[str, float], idf: Dict[str, float]) -> Vector:
    """Multiply TF by IDF for words present in both maps."""
    return {w: tf_val * idf.get(w, 1.0) for w, tf_val in tf.items()}


def _l2_norm(vec: Vector) -> float:
    """Euclidean norm of a sparse vector."""
    return math.sqrt(sum(v * v for v in vec.values()))


def _cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two sparse vectors."""
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Dot product — iterate over the smaller vector
    if len(a) > len(b):
        a, b = b, a
    dot = sum(val * b.get(w, 0.0) for w, val in a.items())
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Topic pseudo-document builder
# ---------------------------------------------------------------------------

def _topic_pseudo_doc(
    lda_result : LDAResult,
    topic_id   : int,
    top_n      : int = 15,
) -> List[str]:
    """
    Build a pseudo-document for a topic by repeating each of its
    top words proportionally to its probability.
    """
    words_probs = lda_result.model.show_topic(topic_id, topn=top_n)
    pseudo: List[str] = []
    for word, prob in words_probs:
        repeat = max(1, round(prob * 100))
        pseudo.extend([word] * repeat)
    return pseudo


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tfidf_cosine_scores(
    sentences  : List[str],
    lda_result : LDAResult,
    top_n      : int = 15,
) -> Tuple[ScoreMatrix, List[Vector]]:
    """
    Compute TF-IDF cosine similarity of each sentence against each topic.

    Parameters
    ----------
    sentences  : Original (pre-processed) sentence strings.
    lda_result : LDAResult from lda_topic_modelling.run_lda().
    top_n      : Number of top words per topic for pseudo-document.

    Returns
    -------
    (score_matrix, sentence_vectors)
      score_matrix     : List[List[float]] — shape [n_sentences × K]
      sentence_vectors : TF-IDF vector per sentence (for inspection).
    """
    K = lda_result.num_topics

    # --- Tokenise all sentences ---
    corpus_tokens = [_tokenise(s) for s in sentences]

    # --- Compute IDF over all sentences ---
    idf = _compute_idf(corpus_tokens)

    # --- Build TF-IDF vectors for each sentence ---
    sentence_vectors: List[Vector] = []
    for tokens in corpus_tokens:
        tf  = _compute_tf(tokens)
        vec = _tfidf_vector(tf, idf)
        sentence_vectors.append(vec)

    # --- Build TF-IDF vectors for each topic pseudo-doc ---
    topic_vectors: List[Vector] = []
    for topic_id in range(K):
        pseudo_tokens = _topic_pseudo_doc(lda_result, topic_id, top_n)
        tf  = _compute_tf(pseudo_tokens)
        vec = _tfidf_vector(tf, idf)
        topic_vectors.append(vec)

    # --- Compute cosine similarity matrix ---
    score_matrix: ScoreMatrix = []
    for sent_vec in sentence_vectors:
        row = [_cosine(sent_vec, topic_vec) for topic_vec in topic_vectors]
        score_matrix.append(row)

    return score_matrix, sentence_vectors


def print_score_matrix(
    sentences    : List[str],
    score_matrix : ScoreMatrix,
    max_preview  : int = 60,
) -> None:
    """Pretty-print the sentence × topic score matrix."""
    K = len(score_matrix[0]) if score_matrix else 0
    header = "  {:>3}  {:<{w}}  {}".format(
        "IDX", "SENTENCE", "  ".join(f"T{k}" for k in range(K)),
        w=max_preview,
    )
    print(f"\n{'='*len(header)}")
    print("TF-IDF Cosine Score Matrix")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for i, (sent, row) in enumerate(zip(sentences, score_matrix)):
        preview = sent[:max_preview].ljust(max_preview)
        scores  = "  ".join(f"{s:.3f}" for s in row)
        print(f"  {i:>3}  {preview}  {scores}")
    print()
