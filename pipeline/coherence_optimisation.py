# =============================================================
# coherence_optimisation.py
#
# Stage: Coherence Optimisation
# Computes a UMass-style coherence score for an LDA result
# WITHOUT using gensim's CoherenceModel — implemented from first
# principles using co-occurrence counts over the training corpus.
#
# UMass coherence formula (Mimno et al., 2011):
#
#   C(k) = sum_{m=2}^{M} sum_{l=1}^{m-1}
#               log( D(v_m, v_l) + 1 ) / D(v_l)
#
# where M is the number of top words per topic, D(v) is the
# document frequency of word v, and D(v_m, v_l) is the number
# of documents that contain BOTH v_m and v_l.
# =============================================================
'''
from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Tuple

from lda_topic_modelling import LDAResult


# ---------------------------------------------------------------------------
# Internal: build co-occurrence index from the BoW corpus
# ---------------------------------------------------------------------------

def _build_cooccurrence(
    corpus    : list,          # Gensim BoW corpus
    top_n     : int = 10,      # top words considered per topic
) -> Tuple[defaultdict, defaultdict]:
    """
    Returns
    -------
    doc_freq   : {word_id -> document count}
    co_doc_freq: {(word_id_a, word_id_b) -> document count}  (a < b)
    """
    doc_freq   : defaultdict = defaultdict(int)
    co_doc_freq: defaultdict = defaultdict(int)

    for doc in corpus:
        word_ids = [wid for wid, _ in doc]
        word_set = set(word_ids)

        # Document frequency
        for wid in word_set:
            doc_freq[wid] += 1

        # Co-document frequency (all pairs)
        word_list = sorted(word_set)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                pair = (word_list[i], word_list[j])
                co_doc_freq[pair] += 1

    return doc_freq, co_doc_freq


# ---------------------------------------------------------------------------
# UMass coherence for a single topic
# ---------------------------------------------------------------------------

def _umass_single_topic(
    top_word_ids  : List[int],
    doc_freq      : defaultdict,
    co_doc_freq   : defaultdict,
    epsilon       : float = 1.0,
) -> float:
    """
    Computes UMass coherence score for one topic given its top-word ids.
    epsilon is the smoothing constant (default=1 per original paper).
    """
    score = 0.0
    M = len(top_word_ids)
    for m in range(1, M):       # m from 1 to M-1
        v_m = top_word_ids[m]
        for l in range(0, m):   # l from 0 to m-1
            v_l = top_word_ids[l]
            pair = (min(v_m, v_l), max(v_m, v_l))
            d_ml = co_doc_freq.get(pair, 0)
            d_l  = doc_freq.get(v_l, 0)
            if d_l == 0:
                continue
            score += math.log((d_ml + epsilon) / d_l)
    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_coherence(
    lda_result : LDAResult,
    top_n      : int   = 10,
    epsilon    : float = 1.0,
) -> Tuple[float, List[float]]:
    """
    Compute UMass coherence for the trained LDA model.

    Parameters
    ----------
    lda_result : LDAResult from lda_topic_modelling.run_lda()
    top_n      : Number of top words per topic to consider.
    epsilon    : Smoothing constant.

    Returns
    -------
    (mean_coherence, per_topic_scores)
      mean_coherence  : float — average across all topics (higher = better).
      per_topic_scores: List[float] — individual topic coherence values.
    """
    model      = lda_result.model
    corpus     = lda_result.corpus
    dictionary = lda_result.dictionary
    K          = lda_result.num_topics

    if not corpus:
        raise ValueError("Corpus is empty — cannot compute coherence.")

    # Build co-occurrence tables
    doc_freq, co_doc_freq = _build_cooccurrence(corpus, top_n)

    per_topic_scores: List[float] = []

    for topic_id in range(K):
        # Get top-N words as (word_string, probability) pairs
        top_words = model.show_topic(topic_id, topn=top_n)
        # Map word strings back to dictionary ids
        top_word_ids = []
        for word_str, _ in top_words:
            wid = dictionary.token2id.get(word_str)
            if wid is not None:
                top_word_ids.append(wid)

        score = _umass_single_topic(top_word_ids, doc_freq, co_doc_freq, epsilon)
        per_topic_scores.append(score)

    mean_coherence = (
        sum(per_topic_scores) / len(per_topic_scores)
        if per_topic_scores else 0.0
    )

    return mean_coherence, per_topic_scores


def print_coherence_report(
    mean_coherence  : float,
    per_topic_scores: List[float],
) -> None:
    """Pretty-print coherence results."""
    print(f"\n{'='*60}")
    print("Coherence Report  (UMass, higher is better)")
    print(f"{'='*60}")
    for i, s in enumerate(per_topic_scores):
        bar = "█" * max(0, int((s + 20) * 2))   # rough visual bar
        print(f"  Topic {i:02d}: {s:8.4f}  {bar}")
    print(f"  {'Mean':>8}: {mean_coherence:8.4f}")
    print()
'''
# =============================================================
# coherence_optimisation.py  — Stage 2
#
# Changes from v1:
#   - Added corpus-size penalty to the UMass score.
#     On small corpora (< 100 sentences), raw UMass keeps
#     improving with higher K because more topics = tighter
#     word co-occurrence within each partition, regardless of
#     whether the topics are semantically meaningful.
#     Penalized score = umass + penalty_weight * (K / n_docs)
#     Since UMass is negative, penalty subtracts from it,
#     making higher K less attractive proportional to corpus size.
#   - penalty_weight defaults to 10.0 (tunable via argument).
# =============================================================

from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Tuple

from pipeline.lda_topic_modelling import LDAResult


def _build_cooccurrence(corpus, top_n=10):
    doc_freq    = defaultdict(int)
    co_doc_freq = defaultdict(int)
    for doc in corpus:
        word_ids = [wid for wid, _ in doc]
        word_set = set(word_ids)
        for wid in word_set:
            doc_freq[wid] += 1
        word_list = sorted(word_set)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                co_doc_freq[(word_list[i], word_list[j])] += 1
    return doc_freq, co_doc_freq


def _umass_single_topic(top_word_ids, doc_freq, co_doc_freq, epsilon=1.0):
    score = 0.0
    M = len(top_word_ids)
    for m in range(1, M):
        v_m = top_word_ids[m]
        for l in range(0, m):
            v_l  = top_word_ids[l]
            pair = (min(v_m, v_l), max(v_m, v_l))
            d_ml = co_doc_freq.get(pair, 0)
            d_l  = doc_freq.get(v_l, 0)
            if d_l == 0:
                continue
            score += math.log((d_ml + epsilon) / d_l)
    return score


def compute_coherence(
    lda_result,          # BERTopicResult (or old LDAResult — same interface)
    sentences: list,
    top_n: int = 10,
    epsilon: float = 1.0,
    penalty_weight: float = 0.0,   # kept for call-site compat, ignored
):
    """
    UMass coherence computed directly from sentences.
    Works with both BERTopicResult and the old gensim LDAResult
    as long as result.model.show_topic(i, topn=N) is available.
    """
    import math, re
    from collections import defaultdict

    # Build co-occurrence index from raw sentences
    doc_freq    = defaultdict(int)
    co_doc_freq = defaultdict(int)
    for sentence in sentences:
        words = set(w for w in re.findall(r'[a-z]+', sentence.lower()) if len(w) >= 3)
        for w in words:
            doc_freq[w] += 1
        word_list = sorted(words)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                co_doc_freq[(word_list[i], word_list[j])] += 1

    # UMass score per topic
    per_topic_scores = []
    for topic_id in range(lda_result.num_topics):
        top_words = [w for w, _ in lda_result.model.show_topic(topic_id, topn=top_n)]
        score = 0.0
        for m in range(1, len(top_words)):
            for l in range(0, m):
                v_m, v_l = top_words[m], top_words[l]
                pair     = (min(v_m, v_l), max(v_m, v_l))
                d_ml     = co_doc_freq.get(pair, 0)
                d_l      = doc_freq.get(v_l, 0)
                if d_l > 0:
                    score += math.log((d_ml + epsilon) / d_l)
        per_topic_scores.append(score)

    mean = sum(per_topic_scores) / len(per_topic_scores) if per_topic_scores else 0.0
    return mean, per_topic_scores


def print_coherence_report(mean_coherence, per_topic_scores):
    print(f"\n{'='*60}")
    print("Coherence Report  (UMass penalised, higher is better)")
    print(f"{'='*60}")
    for i, s in enumerate(per_topic_scores):
        bar = "█" * max(0, int((s + 20) * 2))
        print(f"  Topic {i:02d}: {s:8.4f}  {bar}")
    print(f"  {'Penalised mean':>14}: {mean_coherence:8.4f}")
    print()