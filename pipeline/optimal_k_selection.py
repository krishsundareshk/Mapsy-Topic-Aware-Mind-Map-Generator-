# =============================================================
# optimal_k_selection.py
#
# Stage: Optimal K* Selection
# Tries every K in [k_min, k_max], trains LDA for each, computes
# UMass coherence (via coherence_optimisation.py), and selects
# the K with the highest coherence score.
#
# No external library is used for the selection logic itself —
# only our own lda_topic_modelling and coherence_optimisation
# modules.
# =============================================================
'''
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple

from lda_topic_modelling import LDAResult, run_lda
from coherence_optimisation import compute_coherence


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

class KSelectionResult(NamedTuple):
    """
    optimal_k      : The chosen K*.
    optimal_result : LDAResult trained with optimal_k.
    k_scores       : List of (k, mean_coherence) for every k tried.
    """
    optimal_k      : int
    optimal_result : LDAResult
    k_scores       : List[Tuple[int, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elbow_k(k_scores: List[Tuple[int, float]]) -> int:
    """
    Secondary elbow-method heuristic: find the K where marginal
    coherence improvement drops below a threshold (knee point).
    If no clear elbow is found, the K with max coherence is returned.

    The elbow is the point with the maximum second derivative
    (curvature) — i.e. where improvement rate falls fastest.
    """
    if len(k_scores) < 3:
        return max(k_scores, key=lambda x: x[1])[0]

    ks     = [ks[0] for ks in k_scores]
    scores = [ks[1] for ks in k_scores]

    # First differences (slopes)
    d1 = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
    # Second differences (curvature)
    d2 = [d1[i+1] - d1[i] for i in range(len(d1)-1)]

    if not d2:
        return max(k_scores, key=lambda x: x[1])[0]

    # Elbow = index of maximum curvature change
    # (most negative second derivative = sharpest bend downward)
    elbow_idx = d2.index(min(d2)) + 1   # +1 offset for d1 shift
    return ks[elbow_idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_optimal_k(
    sentences      : List[str],
    k_min          : int = 2,
    k_max          : int = 8,
    top_n_coherence: int = 10,
    passes         : int = 15,
    iterations     : int = 100,
    random_state   : int = 42,
    strategy       : str = "max_coherence",
    verbose        : bool = True,
) -> KSelectionResult:
    """
    Train LDA for K ∈ [k_min, k_max] and select the best K.

    Parameters
    ----------
    sentences       : Pre-processed sentence strings.
    k_min, k_max    : Range of K values to explore (inclusive).
    top_n_coherence : Top words per topic for coherence computation.
    passes          : LDA training passes for each K.
    iterations      : LDA E-step iterations.
    random_state    : Reproducibility seed.
    strategy        : "max_coherence" (pick peak) or "elbow"
                      (knee-point of coherence curve).
    verbose         : Print progress table.

    Returns
    -------
    KSelectionResult
    """
    if k_min < 2:
        raise ValueError("k_min must be >= 2.")
    if k_max < k_min:
        raise ValueError("k_max must be >= k_min.")

    # Cap k_max to avoid requesting more topics than sentences
    effective_k_max = min(k_max, len(sentences) - 1)
    if effective_k_max < k_min:
        raise ValueError(
            f"Not enough sentences ({len(sentences)}) to try "
            f"k_min={k_min}. Provide more input data."
        )

    k_scores     : List[Tuple[int, float]] = []
    k_results    : List[Tuple[int, LDAResult]] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimal K Selection  (strategy='{strategy}')")
        print(f"Trying K = {k_min} … {effective_k_max}")
        print(f"{'='*60}")
        print(f"  {'K':>3}  {'Mean Coherence':>16}  {'Status'}")
        print(f"  {'-'*3}  {'-'*16}  {'-'*20}")

    for k in range(k_min, effective_k_max + 1):
        try:
            lda_res = run_lda(
                sentences    = sentences,
                num_topics   = k,
                passes       = passes,
                iterations   = iterations,
                random_state = random_state,
            )
            mean_coh, _ = compute_coherence(lda_res, top_n=top_n_coherence)
            k_scores.append((k, mean_coh))
            k_results.append((k, lda_res))

            if verbose:
                bar = "+" * max(0, int((mean_coh + 30) * 0.5))
                print(f"  {k:>3}  {mean_coh:>16.4f}  {bar}")

        except Exception as exc:
            if verbose:
                print(f"  {k:>3}  {'ERROR':>16}  {exc}")

    if not k_scores:
        raise RuntimeError("No K value produced a valid LDA model.")

    # --- Select K* ---
    if strategy == "elbow":
        best_k = _elbow_k(k_scores)
    else:  # default: max coherence
        best_k = max(k_scores, key=lambda x: x[1])[0]

    # Retrieve the corresponding result
    best_result = next(r for k, r in k_results if k == best_k)

    if verbose:
        print(f"\n  → Selected K* = {best_k}  "
              f"(coherence = {next(s for k, s in k_scores if k == best_k):.4f})\n")

    return KSelectionResult(
        optimal_k      = best_k,
        optimal_result = best_result,
        k_scores       = k_scores,
    )


def print_k_scores(k_scores: List[Tuple[int, float]]) -> None:
    """Display the coherence curve."""
    print("\nK vs Coherence:")
    for k, score in k_scores:
        bar = "█" * max(0, int((score + 30) * 1))
        print(f"  K={k:>2}: {score:8.4f}  {bar}")
'''
# =============================================================
# optimal_k_selection.py  — Stage 3
#
# Changes from v1:
#   - Added corpus-size guard: effective k_max is capped at
#     len(sentences) // MIN_DOCS_PER_TOPIC (default=8).
#     On 30 sentences → cap = 30//8 = 3.  Prevents LDA from
#     creating micro-topics with only 4-5 sentences each, which
#     always score better on raw UMass but are semantically empty.
#   - k_selection now uses the PENALISED coherence score from
#     coherence_optimisation (not the raw UMass).
#   - "elbow" strategy replaced with "penalised_max" as default:
#     pick the K with the highest penalised score.
#   - Added print of both raw and penalised scores in verbose mode.
# =============================================================

from __future__ import annotations
from unittest import result
import numpy as np
from typing import List, NamedTuple, Optional, Tuple

from pipeline.lda_topic_modelling import LDAResult, run_lda
from pipeline.coherence_optimisation import compute_coherence

MIN_DOCS_PER_TOPIC = 8   # rule-of-thumb: at least 8 sentences per topic


class KSelectionResult(NamedTuple):
    optimal_k      : int
    optimal_result : LDAResult
    k_scores       : List[Tuple[int, float]]   # (k, penalised_coherence)


def select_optimal_k(
    sentences       : List[str],
    embeddings      : Optional[np.ndarray] = None,
    k_min           : int   = 2,
    k_max           : int   = 8,
    top_n_coherence : int   = 10,
    passes          : int   = 20,
    iterations      : int   = 200,
    random_state    : int   = 42,
    strategy        : str   = "penalised_max",
    penalty_weight  : float = 10.0,
    verbose         : bool  = True,
) -> KSelectionResult:
    # --- Corpus-size guard ---
    # Cap k_max so each topic has at least MIN_DOCS_PER_TOPIC sentences.
    size_cap       = max(k_min, len(sentences) // MIN_DOCS_PER_TOPIC)
    effective_k_max = min(k_max, size_cap)

    if effective_k_max < k_min:
        raise ValueError(
            f"Corpus too small ({len(sentences)} sentences) for k_min={k_min}. "
            f"Need at least {k_min * MIN_DOCS_PER_TOPIC} sentences."
        )

    k_scores  : List[Tuple[int, float]] = []
    k_results : List[Tuple[int, LDAResult]] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimal K Selection  (strategy='{strategy}')")
        print(f"Corpus size: {len(sentences)} sentences  |  "
              f"Size cap: K ≤ {effective_k_max}  (was k_max={k_max})")
        print(f"{'='*60}")
        print(f"  {'K':>3}  {'Raw UMass':>12}  {'Penalised':>12}  {'Status'}")
        print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*20}")

    for k in range(k_min, effective_k_max + 1):
        try:
            lda_res = run_lda(
                sentences    = sentences,
                num_topics   = k,
                embeddings   = embeddings,
                passes       = passes,
                iterations   = iterations,
                random_state = random_state,
            )
            penalised_coh, per_topic = compute_coherence(
                lda_res,
                sentences=sentences,
                top_n          = top_n_coherence,
                penalty_weight = penalty_weight,
            )
            raw_mean = sum(per_topic) / len(per_topic)

            k_scores.append((k, penalised_coh))
            k_results.append((k, lda_res))
            print(f"  [DEBUG] K={k} → num_topics={result.num_topics}, ok")
            if verbose:
                bar = "+" * max(0, int((penalised_coh + 50) * 0.5))
                print(f"  {k:>3}  {raw_mean:>12.4f}  {penalised_coh:>12.4f}  {bar}")

        except Exception as exc:
            if verbose:
                print(f"  {k:>3}  {'ERROR':>12}  {'':>12}  {exc}")

    if not k_scores:
        raise RuntimeError("No valid K found.")

    best_k      = max(k_scores, key=lambda x: x[1])[0]
    best_result = next(r for k, r in k_results if k == best_k)
    best_score  = next(s for k, s in k_scores if k == best_k)

    if verbose:
        print(f"\n  → Selected K* = {best_k}  (penalised coherence = {best_score:.4f})\n")

    return KSelectionResult(
        optimal_k      = best_k,
        optimal_result = best_result,
        k_scores       = k_scores,
    )


def print_k_scores(k_scores):
    print("\nK vs Penalised Coherence:")
    for k, score in k_scores:
        bar = "█" * max(0, int((score + 50) * 1))
        print(f"  K={k:>2}: {score:8.4f}  {bar}")