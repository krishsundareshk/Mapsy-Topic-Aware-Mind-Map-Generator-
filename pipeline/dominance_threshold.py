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
'''
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
'''
# =============================================================
# dominance_threshold.py  — Stage 5
#
# Changes from v1:
#   - auto_percentile default lowered from 0.35 → 0.15.
#     Previously the 35th-pct threshold cut 30% of sentences.
#     With all fusion scores bunched in 0.30-0.36, even correct
#     assignments fell below the cutoff.
#   - Added FALLBACK ASSIGNMENT: sentences that don't clear the
#     confidence threshold are no longer dropped as "ambiguous".
#     Instead, they are assigned to the topic with the highest
#     fusion score and flagged as "soft" (low-confidence).
#     This guarantees 100% sentence coverage.
#   - ClusteringResult now carries a 'soft_assignments' set
#     so downstream stages can distinguish confident vs soft.
#   - 'ambiguous' list is kept for backward compatibility but
#     will always be empty when use_fallback=True (default).
# =============================================================

from __future__ import annotations
import math
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from pipeline.lda_topic_modelling import LDAResult
from pipeline.tfidf_cosine_scoring import ScoreMatrix


class ClusteringResult(NamedTuple):
    groups          : Dict[int, List[int]]   # topic_id → [sentence indices]
    ambiguous       : List[int]              # always empty when use_fallback=True
    assignments     : List[Optional[int]]    # topic_id per sentence (never None)
    fusion_matrix   : List[List[float]]
    threshold_used  : float
    soft_assignments: Set[int]               # sentence indices assigned by fallback


def _softmax(values):
    m   = max(values)
    exp = [math.exp(v - m) for v in values]
    s   = sum(exp)
    return [e / s for e in exp]


def _fuse_scores(lda_doc_topics, cosine_row, K, lda_weight=0.4, cosine_weight=0.6):
    # FIX: default weights flipped vs v1 (lda 0.6 → 0.4, cosine 0.4 → 0.6).
    # LDA is noisy on small corpora; TF-IDF cosine is more reliable.
    lda_vec = [0.0] * K
    for topic_id, prob in lda_doc_topics:
        if 0 <= topic_id < K:
            lda_vec[topic_id] = prob
    lda_norm    = _softmax(lda_vec)    if sum(lda_vec) > 0    else lda_vec
    cosine_norm = _softmax(cosine_row) if sum(cosine_row) > 0 else cosine_row
    return [
        lda_weight * lda_norm[k] + cosine_weight * cosine_norm[k]
        for k in range(K)
    ]


def _auto_threshold(fusion_matrix, percentile=0.15):
    # FIX: percentile lowered from 0.35 → 0.15
    max_scores    = [max(row) for row in fusion_matrix]
    sorted_scores = sorted(max_scores)
    idx = max(0, int(len(sorted_scores) * percentile) - 1)
    return sorted_scores[idx]


def apply_dominance_threshold(
    sentences        : List[str],
    lda_result       : LDAResult,
    score_matrix     : ScoreMatrix,
    threshold        : Optional[float] = None,
    lda_weight       : float = 0.4,     # FIX: was 0.6
    cosine_weight    : float = 0.6,     # FIX: was 0.4
    auto_percentile  : float = 0.15,    # FIX: was 0.35
    use_fallback     : bool  = True,    # NEW: assign low-confidence sentences too
) -> ClusteringResult:
    K           = lda_result.num_topics
    n_sentences = len(sentences)
    doc_topics  = lda_result.doc_topics

    # Build fusion matrix
    fusion_matrix = []
    for i in range(n_sentences):
        fused = _fuse_scores(
            lda_doc_topics = doc_topics[i],
            cosine_row     = score_matrix[i],
            K              = K,
            lda_weight     = lda_weight,
            cosine_weight  = cosine_weight,
        )
        fusion_matrix.append(fused)

    if threshold is None:
        threshold = _auto_threshold(fusion_matrix, percentile=auto_percentile)

    groups      : Dict[int, List[int]] = {k: [] for k in range(K)}
    ambiguous   : List[int]            = []
    assignments : List[Optional[int]]  = []
    soft_set    : Set[int]             = set()

    for i, fused in enumerate(fusion_matrix):
        best_topic = max(range(K), key=lambda k: fused[k])
        best_score = fused[best_topic]

        if best_score >= threshold:
            # Confident assignment
            groups[best_topic].append(i)
            assignments.append(best_topic)
        elif use_fallback:
            # Soft assignment — below threshold but assign to best anyway
            groups[best_topic].append(i)
            assignments.append(best_topic)
            soft_set.add(i)
        else:
            ambiguous.append(i)
            assignments.append(None)

    return ClusteringResult(
        groups           = groups,
        ambiguous        = ambiguous,
        assignments      = assignments,
        fusion_matrix    = fusion_matrix,
        threshold_used   = threshold,
        soft_assignments = soft_set,
    )


def print_clustering_result(result, sentences, max_len=80):
    K = len(result.groups)
    print(f"\n{'='*70}")
    print(f"Clustering Result  (threshold={result.threshold_used:.4f}, "
          f"soft={len(result.soft_assignments)} sentences)")
    print(f"{'='*70}")
    for topic_id in range(K):
        members = result.groups[topic_id]
        print(f"\n  ── Topic {topic_id}  ({len(members)} sentence(s)) ──")
        for idx in members:
            tag     = " [soft]" if idx in result.soft_assignments else ""
            preview = sentences[idx][:max_len]
            score   = result.fusion_matrix[idx][topic_id]
            print(f"    [{idx:>02}]{tag} (score={score:.3f}) {preview}")
    if result.ambiguous:
        print(f"\n  ── Ambiguous  ({len(result.ambiguous)}) ──")
        for idx in result.ambiguous:
            print(f"    [{idx:>02}] {sentences[idx][:max_len]}")
    print()