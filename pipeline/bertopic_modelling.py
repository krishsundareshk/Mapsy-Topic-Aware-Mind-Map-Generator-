"""
pipeline/bertopic_modelling.py
==============================
BERTopic-based topic modelling — drop-in replacement for lda_topic_modelling.py.

Public API kept identical to lda_topic_modelling.py:
  • run_lda(sentences, num_topics, ...) → BERTopicResult
  • print_topics(result, top_n)
  • LDAResult = BERTopicResult  (alias so all existing imports keep working)

Additional helper:
  • compute_embeddings(sentences, ...) → np.ndarray
    Call this ONCE in the orchestrator and pass the result to every
    run_lda() call to avoid re-encoding for each candidate K.

Install:
  pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple, NamedTuple, Any, Optional

import numpy as np

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.WARNING)


class _ModelWrapper:
    def __init__(self, bertopic_model: Any, topic_ids: List[int]) -> None:
        self.bertopic_model = bertopic_model
        self._topic_ids     = topic_ids
        self.num_topics     = len(topic_ids)

    def show_topic(self, internal_id: int, topn: int = 10) -> List[Tuple[str, float]]:
        if internal_id >= len(self._topic_ids):
            return []
        real_id = self._topic_ids[internal_id]
        words   = self.bertopic_model.get_topic(real_id)
        if not words:
            return []
        words = words[:topn]
        total = sum(max(0.0, s) for _, s in words) or 1.0
        return [(w, max(0.0, s) / total) for w, s in words]


class BERTopicResult(NamedTuple):
    model          : _ModelWrapper
    corpus         : list
    dictionary     : Any
    doc_topics     : List[List[Tuple[int, float]]]
    num_topics     : int
    sentences      : List[List[str]]
    bertopic_model : Any


LDAResult = BERTopicResult


def _tokenise(sentence: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z]+", sentence.lower()) if len(t) >= 3]


def compute_embeddings(
    sentences       : List[str],
    embedding_model : str  = "all-MiniLM-L6-v2",
    verbose         : bool = False,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model      = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        sentences,
        show_progress_bar = verbose,
        convert_to_numpy  = True,
    )
    return embeddings


def run_lda(
    sentences      : List[str],
    num_topics     : int                  = 3,
    passes         : int                  = 20,
    iterations     : int                  = 200,
    random_state   : int                  = 42,
    alpha          : str                  = "symmetric",
    eta            : str                  = "symmetric",
    embedding_model: str                  = "all-MiniLM-L6-v2",
    embeddings     : Optional[np.ndarray] = None,
    min_topic_size : int                  = 2,
) -> BERTopicResult:
    from bertopic import BERTopic
    from sklearn.cluster import KMeans
    from umap import UMAP

    if not sentences:
        raise ValueError("sentences list is empty.")
    if len(sentences) < num_topics:
        raise ValueError(
            f"Only {len(sentences)} sentences but num_topics={num_topics}. "
            "Reduce num_topics or provide more text."
        )

    n_samples  = len(sentences)
    num_topics = min(num_topics, n_samples)   # safety cap

    UMAP_MIN = max(15, num_topics * 3)
    if n_samples < UMAP_MIN:
        from bertopic.dimensionality import BaseDimensionalityReduction
        umap_model: Any = BaseDimensionalityReduction()
    else:
        n_components = min(5, n_samples - 2)
        n_neighbors  = min(15, n_samples - 1)
        umap_model = UMAP(
            n_components = n_components,
            n_neighbors  = n_neighbors,
            min_dist     = 0.0,
            metric       = "cosine",
            random_state = random_state,
        )

    cluster_model = KMeans(
        n_clusters   = num_topics,
        random_state = random_state,
        n_init       = 10,
    )

    # Pass None as embedding_model when embeddings are pre-supplied
    # to prevent BERTopic from reloading the SentenceTransformer unnecessarily
    topic_model = BERTopic(
        embedding_model         = embedding_model if embeddings is None else None,
        umap_model              = umap_model,
        hdbscan_model           = cluster_model,
        calculate_probabilities = True,
        verbose                 = False,
    )

    topics_assigned, probs = topic_model.fit_transform(
        sentences, embeddings=embeddings
    )

    valid_ids = sorted({t for t in topics_assigned if t != -1})
    if not valid_ids:
        valid_ids = list(range(num_topics))
    id_map   = {real: i for i, real in enumerate(valid_ids)}
    actual_k = len(valid_ids)

    doc_topics: List[List[Tuple[int, float]]] = []
    for i in range(n_samples):
        if (
            probs is not None
            and i < len(probs)
            and len(probs[i]) == actual_k
        ):
            row   = [max(0.0, float(p)) for p in probs[i]]
            total = sum(row) or 1.0
            dist  = [(j, row[j] / total) for j in range(actual_k)]
        else:
            assigned = id_map.get(int(topics_assigned[i]), 0)
            dist = [(j, 1.0 if j == assigned else 0.0) for j in range(actual_k)]
        doc_topics.append(dist)

    wrapper   = _ModelWrapper(topic_model, valid_ids)
    tokenised = [_tokenise(s) for s in sentences]

    return BERTopicResult(
        model          = wrapper,
        corpus         = [],
        dictionary     = None,
        doc_topics     = doc_topics,
        num_topics     = actual_k,
        sentences      = tokenised,
        bertopic_model = topic_model,
    )


def print_topics(result: BERTopicResult, top_n: int = 10) -> None:
    print(f"\n{'='*60}")
    print(f"BERTopic Topics  (K = {result.num_topics})")
    print(f"{'='*60}")
    for i in range(result.num_topics):
        terms = result.model.show_topic(i, topn=top_n)
        words = ", ".join(f"{w}({p:.3f})" for w, p in terms)
        print(f"  Topic {i}: {words}")
    print()