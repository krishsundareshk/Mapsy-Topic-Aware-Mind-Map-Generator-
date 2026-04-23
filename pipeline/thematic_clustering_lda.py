# =============================================================
# thematic_clustering.py
#
# Main Orchestrator — Thematic Clustering Stage
#
# Pipeline order (mirrors the architecture diagram):
#   1. lda_topic_modelling    → LDAResult
#   2. coherence_optimisation → coherence score
#   3. optimal_k_selection    → K* and best LDAResult
#   4. tfidf_cosine_scoring   → ScoreMatrix
#   5. dominance_threshold    → ClusteringResult
#
# Inputs  : a list of clean sentences (output of C preprocessing)
# Outputs : ClusteringResult — sentence groups per topic (K* groups)
#
# Usage (standalone):
#   python thematic_clustering.py <clean_sentences_file>
# =============================================================
'''
from __future__ import annotations

import sys
import os
from typing import Dict, List, NamedTuple, Optional

# --- local modules ---
from lda_topic_modelling    import LDAResult, run_lda, print_topics
from coherence_optimisation import compute_coherence, print_coherence_report
from optimal_k_selection    import KSelectionResult, select_optimal_k, print_k_scores
from tfidf_cosine_scoring   import ScoreMatrix, compute_tfidf_cosine_scores, print_score_matrix
from dominance_threshold    import ClusteringResult, apply_dominance_threshold, print_clustering_result


# ---------------------------------------------------------------------------
# Data contract — full pipeline output
# ---------------------------------------------------------------------------

class ThematicClusteringResult(NamedTuple):
    """
    All intermediate and final results produced by the pipeline.

    optimal_k        : The chosen K*.
    lda_result       : Trained LDA model for K*.
    mean_coherence   : Final model coherence score.
    score_matrix     : TF-IDF cosine score matrix [n_sent × K*].
    clustering       : ClusteringResult with sentence → topic assignments.
    sentences        : The input sentences (for reference).
    sentence_groups  : Dict[topic_id -> List[str]] — the final output.
    """
    optimal_k       : int
    lda_result      : LDAResult
    mean_coherence  : float
    score_matrix    : ScoreMatrix
    clustering      : ClusteringResult
    sentences       : List[str]
    sentence_groups : Dict[int, List[str]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ThematicConfig:
    """Single place to tune all pipeline hyper-parameters."""
    def __init__(
        self,
        k_min              : int   = 2,
        k_max              : int   = 7,
        lda_passes         : int   = 20,
        lda_iterations     : int   = 150,
        random_state       : int   = 42,
        k_strategy         : str   = "max_coherence",  # or "elbow"
        top_n_words        : int   = 12,
        coherence_top_n    : int   = 10,
        lda_weight         : float = 0.6,
        cosine_weight      : float = 0.4,
        dominance_threshold: Optional[float] = None,   # None = auto
        auto_percentile    : float = 0.35,
        verbose            : bool  = True,
    ):
        self.k_min               = k_min
        self.k_max               = k_max
        self.lda_passes          = lda_passes
        self.lda_iterations      = lda_iterations
        self.random_state        = random_state
        self.k_strategy          = k_strategy
        self.top_n_words         = top_n_words
        self.coherence_top_n     = coherence_top_n
        self.lda_weight          = lda_weight
        self.cosine_weight       = cosine_weight
        self.dominance_threshold = dominance_threshold
        self.auto_percentile     = auto_percentile
        self.verbose             = verbose


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_thematic_clustering(
    sentences : List[str],
    config    : Optional[ThematicConfig] = None,
) -> ThematicClusteringResult:
    """
    Execute the full thematic clustering pipeline.

    Parameters
    ----------
    sentences : Clean sentence strings from the preprocessing stage.
    config    : Pipeline configuration (None = sensible defaults).

    Returns
    -------
    ThematicClusteringResult — all intermediate and final data.
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    cfg = config or ThematicConfig()
    vb  = cfg.verbose

    # ----------------------------------------------------------------
    # Stage 1 — LDA Topic Modelling (initial run at k_min for warm-up)
    # ----------------------------------------------------------------
    if vb: _section("1 / 5  LDA Topic Modelling")

    # We first run an initial LDA at k_min to provide something for
    # the coherence optimisation module; the optimal K search below
    # will train fresh models for each K independently.
    initial_lda = run_lda(
        sentences    = sentences,
        num_topics   = cfg.k_min,
        passes       = cfg.lda_passes,
        iterations   = cfg.lda_iterations,
        random_state = cfg.random_state,
    )
    if vb: print_topics(initial_lda, top_n=cfg.top_n_words)

    # ----------------------------------------------------------------
    # Stage 2 — Coherence Optimisation (score the initial model)
    # ----------------------------------------------------------------
    if vb: _section("2 / 5  Coherence Optimisation")

    mean_coh_initial, per_topic_initial = compute_coherence(
        initial_lda, top_n=cfg.coherence_top_n
    )
    if vb: print_coherence_report(mean_coh_initial, per_topic_initial)

    # ----------------------------------------------------------------
    # Stage 3 — Optimal K* Selection
    # ----------------------------------------------------------------
    if vb: _section("3 / 5  Optimal K* Selection")

    k_result: KSelectionResult = select_optimal_k(
        sentences       = sentences,
        k_min           = cfg.k_min,
        k_max           = cfg.k_max,
        top_n_coherence = cfg.coherence_top_n,
        passes          = cfg.lda_passes,
        iterations      = cfg.lda_iterations,
        random_state    = cfg.random_state,
        strategy        = cfg.k_strategy,
        verbose         = vb,
    )

    optimal_k      = k_result.optimal_k
    best_lda       = k_result.optimal_result
    mean_coherence = next(s for k, s in k_result.k_scores if k == optimal_k)

    if vb:
        print_topics(best_lda, top_n=cfg.top_n_words)
        _, per_topic = compute_coherence(best_lda, top_n=cfg.coherence_top_n)
        print_coherence_report(mean_coherence, per_topic)

    # ----------------------------------------------------------------
    # Stage 4 — TF-IDF Cosine Scoring
    # ----------------------------------------------------------------
    if vb: _section("4 / 5  TF-IDF Cosine Scoring")

    score_matrix, _ = compute_tfidf_cosine_scores(
        sentences  = sentences,
        lda_result = best_lda,
        top_n      = cfg.top_n_words,
    )
    if vb: print_score_matrix(sentences, score_matrix)

    # ----------------------------------------------------------------
    # Stage 5 — Dominance Threshold
    # ----------------------------------------------------------------
    if vb: _section("5 / 5  Dominance Threshold")

    clustering: ClusteringResult = apply_dominance_threshold(
        sentences       = sentences,
        lda_result      = best_lda,
        score_matrix    = score_matrix,
        threshold       = cfg.dominance_threshold,
        lda_weight      = cfg.lda_weight,
        cosine_weight   = cfg.cosine_weight,
        auto_percentile = cfg.auto_percentile,
    )
    if vb: print_clustering_result(clustering, sentences)

    # ----------------------------------------------------------------
    # Build final sentence_groups (sentence strings, not indices)
    # ----------------------------------------------------------------
    sentence_groups: Dict[int, List[str]] = {}
    for topic_id, indices in clustering.groups.items():
        sentence_groups[topic_id] = [sentences[i] for i in indices]

    return ThematicClusteringResult(
        optimal_k       = optimal_k,
        lda_result      = best_lda,
        mean_coherence  = mean_coherence,
        score_matrix    = score_matrix,
        clustering      = clustering,
        sentences       = sentences,
        sentence_groups = sentence_groups,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'━'*65}")
    print(f"  STAGE: {title}")
    print(f"{'━'*65}")


def load_sentences_from_file(filepath: str) -> List[str]:
    """Read a clean-sentences file (one sentence per line)."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python thematic_clustering.py <clean_sentences_file>")
        print("       (one sentence per line, UTF-8)")
        sys.exit(1)

    input_file = sys.argv[1]
    sentences  = load_sentences_from_file(input_file)

    print(f"\nLoaded {len(sentences)} sentence(s) from '{input_file}'")

    config = ThematicConfig(
        k_min   = 2,
        k_max   = 6,
        verbose = True,
    )

    result = run_thematic_clustering(sentences, config)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"  Optimal K*         : {result.optimal_k}")
    print(f"  Mean coherence     : {result.mean_coherence:.4f}")
    print(f"  Threshold used     : {result.clustering.threshold_used:.4f}")
    print(f"  Ambiguous sentences: {len(result.clustering.ambiguous)}")
    for topic_id, group in result.sentence_groups.items():
        print(f"  Topic {topic_id} sentences  : {len(group)}")
    print()
'''
# =============================================================
# thematic_clustering.py  — Orchestrator  (v2)
#
# Config changes from v1:
#   k_min=2, k_max=4          (was 2,6 — size guard handles cap)
#   lda_passes=20             (same)
#   lda_iterations=200        (was 100 — more stable on small corpus)
#   penalty_weight=10.0       (NEW — coherence corpus-size penalty)
#   lda_weight=0.4            (was 0.6)
#   cosine_weight=0.6         (was 0.4)
#   dominance_threshold=None  (auto)
#   auto_percentile=0.15      (was 0.35)
#   use_fallback=True         (NEW — no sentence left unassigned)
# =============================================================

from __future__ import annotations
import sys, os
from typing import Dict, List, NamedTuple, Optional, Set

from pipeline.lda_topic_modelling    import LDAResult, run_lda, print_topics
from pipeline.coherence_optimisation import compute_coherence, print_coherence_report
from pipeline.optimal_k_selection    import KSelectionResult, select_optimal_k
from pipeline.tfidf_cosine_scoring   import ScoreMatrix, compute_tfidf_cosine_scores, print_score_matrix
from pipeline.dominance_threshold    import ClusteringResult, apply_dominance_threshold, print_clustering_result


class ThematicClusteringResult(NamedTuple):
    optimal_k       : int
    lda_result      : LDAResult
    mean_coherence  : float
    score_matrix    : ScoreMatrix
    clustering      : ClusteringResult
    sentences       : List[str]
    sentence_groups : Dict[int, List[str]]
    soft_indices    : Set[int]


class ThematicConfig:
    def __init__(
        self,
        k_min              : int   = 2,
        k_max              : int   = 4,      # v2: was 6; size guard applies on top
        lda_passes         : int   = 20,
        lda_iterations     : int   = 200,    # v2: was 100
        random_state       : int   = 42,
        k_strategy         : str   = "penalised_max",
        penalty_weight     : float = 10.0,   # v2: NEW — coherence penalty
        top_n_words        : int   = 12,
        coherence_top_n    : int   = 10,
        lda_weight         : float = 0.4,    # v2: was 0.6
        cosine_weight      : float = 0.6,    # v2: was 0.4
        dominance_threshold: Optional[float] = None,
        auto_percentile    : float = 0.15,   # v2: was 0.35
        use_fallback       : bool  = True,   # v2: NEW
        verbose            : bool  = True,
    ):
        self.k_min               = k_min
        self.k_max               = k_max
        self.lda_passes          = lda_passes
        self.lda_iterations      = lda_iterations
        self.random_state        = random_state
        self.k_strategy          = k_strategy
        self.penalty_weight      = penalty_weight
        self.top_n_words         = top_n_words
        self.coherence_top_n     = coherence_top_n
        self.lda_weight          = lda_weight
        self.cosine_weight       = cosine_weight
        self.dominance_threshold = dominance_threshold
        self.auto_percentile     = auto_percentile
        self.use_fallback        = use_fallback
        self.verbose             = verbose


def run_thematic_clustering(
    sentences : List[str],
    config    : Optional[ThematicConfig] = None,
) -> ThematicClusteringResult:
    if not sentences:
        raise ValueError("sentences list is empty.")
    cfg = config or ThematicConfig()
    vb  = cfg.verbose

    # ── Stage 1: LDA (warm-up at k_min) ─────────────────────────
    if vb: _section("1 / 5  LDA Topic Modelling")
    initial_lda = run_lda(
        sentences    = sentences,
        num_topics   = cfg.k_min,
        passes       = cfg.lda_passes,
        iterations   = cfg.lda_iterations,
        random_state = cfg.random_state,
    )
    if vb: print_topics(initial_lda, top_n=cfg.top_n_words)

    # ── Stage 2: Coherence of initial model ──────────────────────
    if vb: _section("2 / 5  Coherence Optimisation")
    mean_coh_init, per_init = compute_coherence(
        initial_lda,
        sentences=sentences,
        top_n          = cfg.coherence_top_n,
        penalty_weight = cfg.penalty_weight,
    )
    if vb: print_coherence_report(mean_coh_init, per_init)

    # ── Stage 3: Optimal K* ───────────────────────────────────────
    if vb: _section("3 / 5  Optimal K* Selection")
    k_result: KSelectionResult = select_optimal_k(
        sentences       = sentences,
        k_min           = cfg.k_min,
        k_max           = cfg.k_max,
        top_n_coherence = cfg.coherence_top_n,
        passes          = cfg.lda_passes,
        iterations      = cfg.lda_iterations,
        random_state    = cfg.random_state,
        strategy        = cfg.k_strategy,
        penalty_weight  = cfg.penalty_weight,
        verbose         = vb,
    )
    optimal_k      = k_result.optimal_k
    best_lda       = k_result.optimal_result
    mean_coherence = next(s for k, s in k_result.k_scores if k == optimal_k)

    if vb:
        print_topics(best_lda, top_n=cfg.top_n_words)
        _, per = compute_coherence(best_lda, sentences=sentences, top_n=cfg.coherence_top_n,
                                   penalty_weight=cfg.penalty_weight)
        print_coherence_report(mean_coherence, per)

    # ── Stage 4: TF-IDF Cosine Scoring ───────────────────────────
    if vb: _section("4 / 5  TF-IDF Cosine Scoring")
    score_matrix, _ = compute_tfidf_cosine_scores(
        sentences  = sentences,
        lda_result = best_lda,
        top_n      = cfg.top_n_words,
    )
    if vb: print_score_matrix(sentences, score_matrix)

    # ── Stage 5: Dominance Threshold ─────────────────────────────
    if vb: _section("5 / 5  Dominance Threshold")
    clustering: ClusteringResult = apply_dominance_threshold(
        sentences       = sentences,
        lda_result      = best_lda,
        score_matrix    = score_matrix,
        threshold       = cfg.dominance_threshold,
        lda_weight      = cfg.lda_weight,
        cosine_weight   = cfg.cosine_weight,
        auto_percentile = cfg.auto_percentile,
        use_fallback    = cfg.use_fallback,
    )
    if vb: print_clustering_result(clustering, sentences)

    # ── Build output ──────────────────────────────────────────────
    sentence_groups: Dict[int, List[str]] = {}
    for topic_id, indices in clustering.groups.items():
        sentence_groups[topic_id] = [sentences[i] for i in indices]

    return ThematicClusteringResult(
        optimal_k       = optimal_k,
        lda_result      = best_lda,
        mean_coherence  = mean_coherence,
        score_matrix    = score_matrix,
        clustering      = clustering,
        sentences       = sentences,
        sentence_groups = sentence_groups,
        soft_indices    = clustering.soft_assignments,
    )


def _section(title):
    print(f"\n{'━'*65}")
    print(f"  STAGE: {title}")
    print(f"{'━'*65}")


def load_sentences_from_file(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python thematic_clustering.py <clean_sentences_file>")
        sys.exit(1)

    sentences = load_sentences_from_file(sys.argv[1])
    print(f"\nLoaded {len(sentences)} sentence(s) from '{sys.argv[1]}'")

    result = run_thematic_clustering(sentences, ThematicConfig(verbose=True))

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"  Optimal K*              : {result.optimal_k}")
    print(f"  Mean penalised coherence: {result.mean_coherence:.4f}")
    print(f"  Threshold used          : {result.clustering.threshold_used:.4f}")
    print(f"  Confident assignments   : {len(result.sentences) - len(result.soft_indices)}")
    print(f"  Soft assignments        : {len(result.soft_indices)}")
    print(f"  Ambiguous (dropped)     : {len(result.clustering.ambiguous)}")
    for topic_id, group in result.sentence_groups.items():
        soft_count = sum(1 for i in result.clustering.groups[topic_id]
                         if i in result.soft_indices)
        print(f"  Topic {topic_id} sentences       : {len(group)}  "
              f"({len(group)-soft_count} confident, {soft_count} soft)")
    print()