# =============================================================
# dominance_threshold.py
#
# Stage: Dominance Threshold
# Assigns each sentence to exactly one topic cluster using a
# combined dominance score that fuses:
#   (a) LDA topic probability (from lda_topic_modelling)
#   (b) TF-IDF cosine similarity (from tfidf_cosine_scoring)
#
# A sentence is assigned to a topic only if its dominant topic
# score clears a minimum threshold.  If it does not clear the
# threshold the sentence is placed in an "ambiguous" bin.
#
# Entirely implemented without external ML libraries.
# =============================================================

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

from lda_topic_modelling import LDAResult
from tfidf_cosine_scoring import ScoreMatrix


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

class ClusteringResult(NamedTuple):
    """
    groups          : Dict[topic_id -> List[sentence_index]]
    ambiguous       : List of sentence indices that did not pass the threshold
    assignments     : List[Optional[int]] — topic_id per sentence (None=ambiguous)
    fusion_matrix   : List[List[float]]  — fused score per sentence per topic
    threshold_used  : float
    """
    groups        : Dict[int, List[int]]
    ambiguous     : List[int]
    assignments   : List[Optional[int]]
    fusion_matrix : List[List[float]]
    threshold_used: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(values: List[float]) -> List[float]:
    """Numerically stable softmax."""
    m   = max(values)
    exp = [math.exp(v - m) for v in values]
    s   = sum(exp)
    return [e / s for e in exp]


def _fuse_scores(
    lda_doc_topics   : List[Tuple[int, float]],
    cosine_row       : List[float],
    K                : int,
    lda_weight       : float = 0.6,
    cosine_weight    : float = 0.4,
) -> List[float]:
    """
    Combine LDA probability and cosine similarity into a single score.

    lda_weight + cosine_weight should equal 1.0.
    Both vectors are softmax-normalised before fusion to put them
    on a comparable scale.
    """
    # Build dense LDA probability vector
    lda_vec = [0.0] * K
    for topic_id, prob in lda_doc_topics:
        if 0 <= topic_id < K:
            lda_vec[topic_id] = prob

    # Normalise
    lda_norm    = _softmax(lda_vec)   if sum(lda_vec) > 0 else lda_vec
    cosine_norm = _softmax(cosine_row) if sum(cosine_row) > 0 else cosine_row

    # Weighted fusion
    fused = [
        lda_weight * lda_norm[k] + cosine_weight * cosine_norm[k]
        for k in range(K)
    ]
    return fused


def _auto_threshold(
    fusion_matrix: List[List[float]],
    percentile   : float = 0.40,
) -> float:
    """
    Set threshold automatically as the `percentile`-th quantile of
    the maximum fused score across all sentences.

    A lower percentile is more permissive (fewer ambiguous).
    """
    max_scores = [max(row) for row in fusion_matrix]
    sorted_scores = sorted(max_scores)
    idx = max(0, int(len(sorted_scores) * percentile) - 1)
    return sorted_scores[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_dominance_threshold(
    sentences        : List[str],
    lda_result       : LDAResult,
    score_matrix     : ScoreMatrix,
    threshold        : Optional[float] = None,
    lda_weight       : float = 0.6,
    cosine_weight    : float = 0.4,
    auto_percentile  : float = 0.35,
) -> ClusteringResult:
    """
    Assign sentences to topic clusters via fused dominance scoring.

    Parameters
    ----------
    sentences       : Pre-processed sentence strings (for count).
    lda_result      : LDAResult from lda_topic_modelling.run_lda().
    score_matrix    : TF-IDF cosine scores (from tfidf_cosine_scoring).
    threshold       : Minimum fused score for a sentence to be assigned.
                      Pass None to compute automatically.
    lda_weight      : Weight for LDA probability in fusion (0–1).
    cosine_weight   : Weight for cosine similarity in fusion (0–1).
    auto_percentile : Percentile used when threshold=None.

    Returns
    -------
    ClusteringResult
    """
    K           = lda_result.num_topics
    n_sentences = len(sentences)
    doc_topics  = lda_result.doc_topics

    if len(doc_topics) != n_sentences:
        raise ValueError(
            f"lda_result.doc_topics has {len(doc_topics)} entries "
            f"but {n_sentences} sentences provided."
        )
    if len(score_matrix) != n_sentences:
        raise ValueError(
            f"score_matrix has {len(score_matrix)} rows "
            f"but {n_sentences} sentences provided."
        )

    # --- Build fusion matrix ---
    fusion_matrix: List[List[float]] = []
    for i in range(n_sentences):
        fused = _fuse_scores(
            lda_doc_topics = doc_topics[i],
            cosine_row     = score_matrix[i],
            K              = K,
            lda_weight     = lda_weight,
            cosine_weight  = cosine_weight,
        )
        fusion_matrix.append(fused)

    # --- Determine threshold ---
    if threshold is None:
        threshold = _auto_threshold(fusion_matrix, percentile=auto_percentile)

    # --- Assign sentences ---
    groups: Dict[int, List[int]] = {k: [] for k in range(K)}
    ambiguous: List[int]         = []
    assignments: List[Optional[int]] = []

    for i, fused in enumerate(fusion_matrix):
        best_topic = max(range(K), key=lambda k: fused[k])
        best_score = fused[best_topic]

        if best_score >= threshold:
            groups[best_topic].append(i)
            assignments.append(best_topic)
        else:
            ambiguous.append(i)
            assignments.append(None)

    return ClusteringResult(
        groups         = groups,
        ambiguous      = ambiguous,
        assignments    = assignments,
        fusion_matrix  = fusion_matrix,
        threshold_used = threshold,
    )


def print_clustering_result(
    result    : ClusteringResult,
    sentences : List[str],
    max_len   : int = 80,
) -> None:
    """Pretty-print the final sentence clusters."""
    K = len(result.groups)
    print(f"\n{'='*70}")
    print(f"Clustering Result  (threshold = {result.threshold_used:.4f})")
    print(f"{'='*70}")

    for topic_id in range(K):
        members = result.groups[topic_id]
        print(f"\n  ── Topic {topic_id}  ({len(members)} sentence(s)) ──")
        for idx in members:
            preview = sentences[idx][:max_len]
            fused   = result.fusion_matrix[idx][topic_id]
            print(f"    [{idx:>02}] (score={fused:.3f}) {preview}")

    if result.ambiguous:
        print(f"\n  ── Ambiguous  ({len(result.ambiguous)} sentence(s)) ──")
        for idx in result.ambiguous:
            print(f"    [{idx:>02}] {sentences[idx][:max_len]}")

    print()
