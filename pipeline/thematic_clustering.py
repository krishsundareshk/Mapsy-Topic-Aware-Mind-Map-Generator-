"""
pipeline/thematic_clustering.py — BERTopic orchestrator (v5)
=============================================================
Replaces the gensim-LDA orchestrator. Output schema is identical
so Phase 3 / 4 / 5 need zero changes.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from pipeline.bertopic_modelling    import BERTopicResult, compute_embeddings, print_topics, run_lda
from pipeline.coherence_optimisation import compute_coherence, print_coherence_report
from pipeline.optimal_k_selection   import KSelectionResult, select_optimal_k
from pipeline.tfidf_cosine_scoring  import ScoreMatrix, compute_tfidf_cosine_scores, print_score_matrix
from pipeline.dominance_threshold   import ClusteringResult, apply_dominance_threshold, print_clustering_result


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class ThematicConfig:
    def __init__(
        self,
        k_min               : int   = 2,
        k_max               : int   = 5,
        random_state        : int   = 42,
        k_strategy          : str   = "max_coherence",
        top_n_words         : int   = 12,
        coherence_top_n     : int   = 10,
        lda_weight          : float = 0.4,
        cosine_weight       : float = 0.6,
        dominance_threshold : Optional[float] = None,
        auto_percentile     : float = 0.15,
        use_fallback        : bool  = True,
        embedding_model     : str   = "all-MiniLM-L6-v2",
        verbose             : bool  = True,
        use_meta_clustering : bool  = True,
        meta_k_max          : int   = 5,
        lda_passes          : int   = 20,
        lda_iterations      : int   = 200,
        penalty_weight      : float = 0.0,
    ) -> None:
        self.k_min               = k_min
        self.k_max               = k_max
        self.random_state        = random_state
        self.k_strategy          = k_strategy
        self.top_n_words         = top_n_words
        self.coherence_top_n     = coherence_top_n
        self.lda_weight          = lda_weight
        self.cosine_weight       = cosine_weight
        self.dominance_threshold = dominance_threshold
        self.auto_percentile     = auto_percentile
        self.use_fallback        = use_fallback
        self.embedding_model     = embedding_model
        self.verbose             = verbose
        self.use_meta_clustering = use_meta_clustering
        self.meta_k_max          = meta_k_max


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

class ThematicClusteringResult(NamedTuple):
    optimal_k             : int
    lda_result            : BERTopicResult
    mean_coherence        : float
    score_matrix          : ScoreMatrix
    clustering            : ClusteringResult
    sentences             : List[str]
    sentence_groups       : Dict[int, List[str]]
    soft_indices          : Set[int]
    meta_topic_map        : Dict[int, int]
    meta_topic_labels     : Dict[int, str]
    meta_sentence_groups  : Dict[int, List[str]]
    micro_sentence_groups : Dict[int, List[str]]


# ─────────────────────────────────────────────────────────────────────────────
# Meta-clustering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_topic_centroids(
    bertopic_result : BERTopicResult,
    embeddings      : np.ndarray,
) -> np.ndarray:
    n_topics   = bertopic_result.num_topics
    doc_topics = bertopic_result.doc_topics
    dim        = embeddings.shape[1]

    centroids = np.zeros((n_topics, dim), dtype=np.float32)
    counts    = np.zeros(n_topics, dtype=np.float32)

    for sent_idx, dist in enumerate(doc_topics):
        dominant_tid = max(dist, key=lambda x: x[1])[0]
        if 0 <= dominant_tid < n_topics:
            centroids[dominant_tid] += embeddings[sent_idx]
            counts[dominant_tid]    += 1.0

    for tid in range(n_topics):
        if counts[tid] > 0:
            centroids[tid] /= counts[tid]

    return centroids


def _elbow_k(
    centroids    : np.ndarray,
    meta_k_max   : int,
    random_state : int,
) -> int:
    from sklearn.cluster import KMeans

    n = len(centroids)
    if n <= 2:
        return n

    k_range = range(2, min(meta_k_max, n) + 1)
    if len(k_range) < 2:
        return 2

    inertias: List[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(centroids)
        inertias.append(float(km.inertia_))

    drops    = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    best_idx = int(np.argmax(drops))
    best_k   = list(k_range)[best_idx + 1]
    return best_k


def _build_meta_topic_labels(
    meta_assignments : np.ndarray,
    bertopic_result  : BERTopicResult,
    top_n            : int = 3,
) -> Dict[int, str]:
    meta_to_micros: Dict[int, List[int]] = defaultdict(list)
    for micro_id, meta_id in enumerate(meta_assignments):
        meta_to_micros[int(meta_id)].append(micro_id)

    labels: Dict[int, str] = {}
    for meta_id, micro_ids in sorted(meta_to_micros.items()):
        word_scores: Dict[str, float] = {}
        for micro_id in micro_ids:
            for word, score in bertopic_result.model.show_topic(micro_id, topn=top_n):
                word_scores[word] = word_scores.get(word, 0.0) + score
        top_words = sorted(word_scores, key=word_scores.__getitem__, reverse=True)[:top_n]
        labels[meta_id] = (
            ' & '.join(w.capitalize() for w in top_words)
            if top_words else f'Meta-Topic {meta_id}'
        )
    return labels


def run_meta_clustering(
    bertopic_result : BERTopicResult,
    embeddings      : np.ndarray,
    meta_k_max      : int  = 5,
    random_state    : int  = 42,
    verbose         : bool = False,
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Cluster micro-topics into meta-topics using KMeans on topic centroids.

    Returns
    -------
    (meta_topic_map, meta_topic_labels)
      meta_topic_map    : {micro_topic_id -> meta_topic_id}
      meta_topic_labels : {meta_topic_id  -> label_string}
    """
    from sklearn.cluster import KMeans

    n_micro = bertopic_result.num_topics

    if n_micro <= 2:
        meta_topic_map    = {i: i for i in range(n_micro)}
        meta_topic_labels = _build_meta_topic_labels(
            np.arange(n_micro), bertopic_result
        )
        if verbose:
            print(f"  [Meta] Only {n_micro} micro-topics — skipping clustering.")
        return meta_topic_map, meta_topic_labels

    centroids = _compute_topic_centroids(bertopic_result, embeddings)
    meta_k    = _elbow_k(centroids, meta_k_max, random_state)

    km = KMeans(n_clusters=meta_k, random_state=random_state, n_init=10)
    meta_assignments = km.fit_predict(centroids)

    meta_topic_map    = {int(micro): int(meta) for micro, meta in enumerate(meta_assignments)}
    meta_topic_labels = _build_meta_topic_labels(meta_assignments, bertopic_result)

    if verbose:
        print(f"\n  Meta-Clustering  ({n_micro} micro → {meta_k} meta-topics)")
        print(f"  {'Micro':>6}  {'Meta':>6}  {'Meta Label':<30}  Top Words")
        print(f"  {'-'*6}  {'-'*6}  {'-'*30}  {'-'*30}")
        for micro_id in range(n_micro):
            meta_id    = meta_topic_map[micro_id]
            meta_label = meta_topic_labels[meta_id]
            micro_words = ', '.join(
                w for w, _ in bertopic_result.model.show_topic(micro_id, topn=4)
            )
            print(f"  {micro_id:>6}  {meta_id:>6}  {meta_label:<30}  {micro_words}")

    return meta_topic_map, meta_topic_labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 1.6 — Merge micro-topic sentence groups into meta-topic groups  (NEW v5)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_to_meta_groups(
    micro_sentence_groups : Dict[int, List[str]],
    meta_topic_map        : Dict[int, int],
    meta_topic_labels     : Dict[int, str],
    verbose               : bool = False,
) -> Dict[int, List[str]]:
    """
    Combine every micro-topic sentence list into its parent meta-topic bucket.

    Preserves insertion order of micro-topics within each meta-group
    (sorted by micro_id for determinism).

    Parameters
    ----------
    micro_sentence_groups : {micro_id -> [sentences]}   (from dominance step)
    meta_topic_map        : {micro_id -> meta_id}
    meta_topic_labels     : {meta_id  -> label}

    Returns
    -------
    meta_groups : {meta_id -> [sentences]}   — de-duplicated, order-preserved
    """
    meta_groups: Dict[int, List[str]] = defaultdict(list)
    seen_per_meta: Dict[int, Set[str]] = defaultdict(set)

    for micro_id in sorted(micro_sentence_groups.keys()):
        meta_id   = meta_topic_map.get(micro_id, micro_id)
        sents     = micro_sentence_groups[micro_id]
        for s in sents:
            if s not in seen_per_meta[meta_id]:
                meta_groups[meta_id].append(s)
                seen_per_meta[meta_id].add(s)

    result = dict(sorted(meta_groups.items()))

    if verbose:
        print(f"\n  Step 1.6 — Merged micro → meta sentence groups:")
        for meta_id, sents in result.items():
            label = meta_topic_labels.get(meta_id, f'Meta {meta_id}')
            micro_ids = [m for m, g in meta_topic_map.items() if g == meta_id]
            print(f"    Meta {meta_id} [{label}]  "
                  f"(micro-topics {micro_ids}): {len(sents)} sentences")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic LDA-result wrapper aligned to meta-topics  (NEW v5)
# ─────────────────────────────────────────────────────────────────────────────

class _MetaModelWrapper:
    """
    Wraps the original BERTopic _ModelWrapper so that show_topic()
    operates on META-topic IDs.

    The top words for a meta-topic are the union of its constituent
    micro-topics' top words, re-ranked by cumulative score.
    """

    def __init__(
        self,
        original_model    : object,
        meta_topic_map    : Dict[int, int],
        n_meta            : int,
        top_n             : int = 15,
    ) -> None:
        self._orig          = original_model
        self._map           = meta_topic_map
        self.num_topics     = n_meta
        self._cache: Dict[int, List[Tuple[str, float]]] = {}
        self._top_n         = top_n
        self._build_cache()

    def _build_cache(self) -> None:
        meta_words: Dict[int, Dict[str, float]] = defaultdict(dict)
        for micro_id, meta_id in self._map.items():
            for word, score in self._orig.show_topic(micro_id, topn=self._top_n):
                meta_words[meta_id][word] = (
                    meta_words[meta_id].get(word, 0.0) + score
                )
        for meta_id, word_scores in meta_words.items():
            ranked = sorted(word_scores.items(), key=lambda x: -x[1])
            total  = sum(s for _, s in ranked) or 1.0
            self._cache[meta_id] = [(w, s / total) for w, s in ranked]

    def show_topic(self, meta_id: int, topn: int = 10) -> List[Tuple[str, float]]:
        return self._cache.get(meta_id, [])[:topn]


def _build_meta_lda_result(
    original_result : BERTopicResult,
    meta_topic_map  : Dict[int, int],
    meta_sentences  : List[str],
    all_sentences   : List[str],
) -> BERTopicResult:
    """
    Construct a BERTopicResult whose num_topics and doc_topics are aligned
    to meta-topic IDs.  This lets TF-IDF scoring and dominance thresholding
    work transparently on meta-topics.

    Parameters
    ----------
    original_result : The K*-topic BERTopicResult from micro-topic selection.
    meta_topic_map  : {micro_id -> meta_id}
    meta_sentences  : Flat list of all sentences (same order as all_sentences).
    all_sentences   : Ditto (used to rebuild doc_topics).

    Returns
    -------
    A new BERTopicResult with:
      num_topics  = number of distinct meta-topics
      model       = _MetaModelWrapper (show_topic works on meta IDs)
      doc_topics  = per-sentence soft assignment over meta-topic IDs
    """
    n_meta = len(set(meta_topic_map.values()))
    wrapper = _MetaModelWrapper(
        original_model = original_result.model,
        meta_topic_map = meta_topic_map,
        n_meta         = n_meta,
    )

    # Re-map per-sentence topic distributions to meta IDs
    new_doc_topics: List[List[Tuple[int, float]]] = []
    for dist in original_result.doc_topics:
        # Aggregate micro-topic probabilities into meta buckets
        meta_probs: Dict[int, float] = defaultdict(float)
        for micro_id, prob in dist:
            meta_id = meta_topic_map.get(micro_id, micro_id)
            meta_probs[meta_id] += prob
        # Normalise
        total = sum(meta_probs.values()) or 1.0
        new_doc_topics.append([
            (meta_id, p / total)
            for meta_id, p in sorted(meta_probs.items())
        ])

    # Build a new BERTopicResult (NamedTuple) with meta-aligned fields
    return BERTopicResult(
        model          = wrapper,
        corpus         = [],
        dictionary     = None,
        doc_topics     = new_doc_topics,
        num_topics     = n_meta,
        sentences      = original_result.sentences,
        bertopic_model = original_result.bertopic_model,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_thematic_clustering(
    sentences : List[str],
    config    : Optional[ThematicConfig] = None,
) -> ThematicClusteringResult:
    """
    Execute the full BERTopic-based thematic clustering pipeline.

    v5 change: after meta-clustering, micro-topics are merged into
    meta-topics and ALL downstream steps operate on meta-topic IDs.

    Parameters
    ----------
    sentences : Clean sentence strings from Phase 1.
    config    : Pipeline configuration (None = sensible defaults).

    Returns
    -------
    ThematicClusteringResult with sentence_groups keyed by meta-topic ID.
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    cfg = config or ThematicConfig()
    vb  = cfg.verbose

    # ── Step 0: Compute embeddings ONCE ──────────────────────────────────────
    if vb:
        _section("0 / 4  Sentence Embeddings  (BERTopic)")
        print(f"  Model  : {cfg.embedding_model}")
        print(f"  Corpus : {len(sentences)} sentences")

    embeddings: np.ndarray = compute_embeddings(
        sentences,
        embedding_model = cfg.embedding_model,
        verbose         = vb,
    )
    if vb:
        print(f"  ✓ Embeddings: {embeddings.shape}")

    # ── Step 1: Optimal K* via coherence ─────────────────────────────────────
    if vb:
        _section("1 / 4  Optimal K* Selection  (BERTopic + UMass coherence)")

    k_result: KSelectionResult = select_optimal_k(
        sentences       = sentences,
        embeddings      = embeddings,
        k_min           = cfg.k_min,
        k_max           = cfg.k_max,
        top_n_coherence = cfg.coherence_top_n,
        random_state    = cfg.random_state,
        strategy        = cfg.k_strategy,
        verbose         = vb,
    )

    optimal_k      = k_result.optimal_k
    best_result    = k_result.optimal_result
    mean_coherence = next(s for k, s in k_result.k_scores if k == optimal_k)

    if vb:
        print_topics(best_result, top_n=cfg.top_n_words)
        _, per = compute_coherence(
            best_result, sentences, top_n=cfg.coherence_top_n
        )
        print_coherence_report(mean_coherence, per)

    # ── Step 1.5: Meta-Clustering ─────────────────────────────────────────────
    meta_topic_map:    Dict[int, int] = {}
    meta_topic_labels: Dict[int, str] = {}
    use_meta = cfg.use_meta_clustering and optimal_k > 2

    if use_meta:
        if vb:
            _section("1.5 / 4  Meta-Clustering  (KMeans on topic centroids)")
        meta_topic_map, meta_topic_labels = run_meta_clustering(
            bertopic_result = best_result,
            embeddings      = embeddings,
            meta_k_max      = cfg.meta_k_max,
            random_state    = cfg.random_state,
            verbose         = vb,
        )
        if vb:
            n_meta = len(set(meta_topic_map.values()))
            print(f"\n  ✓ {optimal_k} micro-topics → {n_meta} meta-topics")
            for meta_id, label in sorted(meta_topic_labels.items()):
                micro_ids = [m for m, g in meta_topic_map.items() if g == meta_id]
                print(f"    Meta {meta_id} [{label}]: micro-topics {micro_ids}")
    else:
        # Identity mapping: every micro-topic IS its own meta-topic
        meta_topic_map    = {i: i for i in range(optimal_k)}
        meta_topic_labels = {
            i: ' & '.join(
                w.capitalize()
                for w, _ in best_result.model.show_topic(i, topn=3)
            )
            for i in range(optimal_k)
        }
        if vb:
            if not cfg.use_meta_clustering:
                print("\n  [Meta-Clustering disabled by config — using micro-topics directly]")
            else:
                print(f"\n  [Meta-Clustering skipped — only {optimal_k} micro-topics]")

    # ── Step 1.6: Build meta-aligned LDA result for TF-IDF / dominance ───────
    # Wrap the original result so that topic IDs now refer to meta-topics.
    if use_meta:
        if vb:
            _section("1.6 / 4  Merging micro → meta topic distributions")
        meta_lda_result = _build_meta_lda_result(
            original_result = best_result,
            meta_topic_map  = meta_topic_map,
            meta_sentences  = sentences,
            all_sentences   = sentences,
        )
        if vb:
            n_meta = meta_lda_result.num_topics
            print(f"  ✓ Meta-LDA result: {n_meta} topics")
            print_topics(meta_lda_result, top_n=cfg.top_n_words)
    else:
        meta_lda_result = best_result   # already micro == meta

    # ── Step 2: TF-IDF Cosine Scoring  (on meta-aligned result) ──────────────
    if vb:
        _section("2 / 4  TF-IDF Cosine Scoring  (meta-topic aligned)")

    score_matrix, _ = compute_tfidf_cosine_scores(
        sentences  = sentences,
        lda_result = meta_lda_result,
        top_n      = cfg.top_n_words,
    )
    if vb:
        print_score_matrix(sentences, score_matrix)

    # ── Step 3: Dominance Threshold  (on meta-aligned result) ────────────────
    if vb:
        _section("3 / 4  Dominance Threshold  (meta-topic aligned)")

    clustering: ClusteringResult = apply_dominance_threshold(
        sentences       = sentences,
        lda_result      = meta_lda_result,
        score_matrix    = score_matrix,
        threshold       = cfg.dominance_threshold,
        lda_weight      = cfg.lda_weight,
        cosine_weight   = cfg.cosine_weight,
        auto_percentile = cfg.auto_percentile,
        use_fallback    = cfg.use_fallback,
    )
    if vb:
        print_clustering_result(clustering, sentences)

    # ── Step 1.6b: Build micro sentence groups (for diagnostics) ─────────────
    # Use original micro-topic best_result for the micro breakdown
    micro_score_matrix, _ = compute_tfidf_cosine_scores(
        sentences  = sentences,
        lda_result = best_result,
        top_n      = cfg.top_n_words,
    )
    micro_clustering = apply_dominance_threshold(
        sentences       = sentences,
        lda_result      = best_result,
        score_matrix    = micro_score_matrix,
        threshold       = cfg.dominance_threshold,
        lda_weight      = cfg.lda_weight,
        cosine_weight   = cfg.cosine_weight,
        auto_percentile = cfg.auto_percentile,
        use_fallback    = cfg.use_fallback,
    )
    micro_sentence_groups: Dict[int, List[str]] = {
        tid: [sentences[i] for i in idxs]
        for tid, idxs in micro_clustering.groups.items()
    }

    # ── Build final meta sentence_groups from dominance step ─────────────────
    # The clustering step already operates on meta IDs (via meta_lda_result),
    # so clustering.groups is keyed by meta_id directly.
    meta_sentence_groups: Dict[int, List[str]] = {
        meta_id: [sentences[i] for i in indices]
        for meta_id, indices in clustering.groups.items()
    }

    # As a safety net, also merge via the explicit meta_topic_map so that any
    # sentence that ended up in a micro-group not remapped by dominance step
    # still reaches the correct meta bucket.
    if use_meta:
        merged = _merge_to_meta_groups(
            micro_sentence_groups = micro_sentence_groups,
            meta_topic_map        = meta_topic_map,
            meta_topic_labels     = meta_topic_labels,
            verbose               = vb,
        )
        # Union: prefer dominance-step assignments, fill gaps from merged
        for meta_id, sents in merged.items():
            if meta_id not in meta_sentence_groups or not meta_sentence_groups[meta_id]:
                meta_sentence_groups[meta_id] = sents
            else:
                # Add any sentences missed by the dominance step
                existing = set(meta_sentence_groups[meta_id])
                for s in sents:
                    if s not in existing:
                        meta_sentence_groups[meta_id].append(s)
                        existing.add(s)

    if vb:
        print(f"\n  ── Final meta-topic sentence groups ──")
        for meta_id in sorted(meta_sentence_groups):
            label = meta_topic_labels.get(meta_id, f'Meta {meta_id}')
            print(f"    Meta {meta_id} [{label}]: "
                  f"{len(meta_sentence_groups[meta_id])} sentences")

    return ThematicClusteringResult(
        optimal_k             = optimal_k,
        lda_result            = meta_lda_result,   # meta-aligned — used by Phase 4
        mean_coherence        = mean_coherence,
        score_matrix          = score_matrix,
        clustering            = clustering,
        sentences             = sentences,
        sentence_groups       = meta_sentence_groups,   # PRIMARY output (meta-keyed)
        soft_indices          = clustering.soft_assignments,
        meta_topic_map        = meta_topic_map,
        meta_topic_labels     = meta_topic_labels,
        meta_sentence_groups  = meta_sentence_groups,
        micro_sentence_groups = micro_sentence_groups,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'━'*65}")
    print(f"  STAGE: {title}")
    print(f"{'━'*65}")


def load_sentences_from_file(filepath: str) -> List[str]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"  Optimal K* (micro)   : {result.optimal_k}")
    print(f"  Mean coherence       : {result.mean_coherence:.4f}")
    n_meta = len(set(result.meta_topic_map.values())) if result.meta_topic_map else result.optimal_k
    print(f"  Meta-topics used     : {n_meta}")
    print(f"  Confident assignments: "
          f"{len(result.sentences) - len(result.soft_indices)}")
    print(f"  Soft assignments     : {len(result.soft_indices)}")
    if result.meta_topic_map:
        for meta_id, label in sorted(result.meta_topic_labels.items()):
            micro_ids = [m for m, g in result.meta_topic_map.items() if g == meta_id]
            print(f"    Meta {meta_id} [{label}]: micro-topics {micro_ids}")
    print(f"\n  Meta-topic sentence groups (downstream input):")
    for meta_id, group in sorted(result.sentence_groups.items()):
        label = result.meta_topic_labels.get(meta_id, f'Meta {meta_id}')
        soft  = sum(
            1 for i in result.clustering.groups.get(meta_id, [])
            if i in result.soft_indices
        )
        print(f"    Meta {meta_id} [{label}]: {len(group)} sentences "
              f"({len(group) - soft} confident, {soft} soft)")
    print()