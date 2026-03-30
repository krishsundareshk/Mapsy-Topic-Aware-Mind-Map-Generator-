# =============================================================
# lda_topic_modelling.py
#
# Stage: LDA Topic Modelling
# Input : List[str]  — clean, pre-processed sentences
# Output: LDAResult  — named-tuple holding the trained model,
#                      Gensim corpus, dictionary, and the
#                      per-sentence topic distribution matrix.
#
# Library used: gensim  (pip install gensim)
# =============================================================

from __future__ import annotations

import re
import logging
from typing import List, Tuple, NamedTuple

from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.WARNING)

# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

class LDAResult(NamedTuple):
    """
    model        : trained Gensim LdaModel
    corpus       : Bag-of-Words corpus (list of sparse vectors)
    dictionary   : Gensim Dictionary (token ↔ id mapping)
    doc_topics   : list[list[(topic_id, probability)]] — one entry per sentence
    num_topics   : K used for this model
    sentences    : the tokenised sentences used for training
    """
    model      : LdaModel
    corpus     : list
    dictionary : corpora.Dictionary
    doc_topics : List[List[Tuple[int, float]]]
    num_topics : int
    sentences  : List[List[str]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STOPWORDS_EXTRA = {
    "also", "would", "could", "should", "may", "might", "shall",
    "must", "can", "will", "one", "two", "three", "four", "five",
    "new", "many", "much", "well", "even", "still", "around",
    "across", "within", "every", "each", "already",
}

def _tokenise(sentence: str) -> List[str]:
    """
    Lowercase, strip non-alpha characters, remove stop-words,
    discard tokens shorter than 3 characters.
    """
    tokens = re.findall(r"[a-zA-Z]+", sentence.lower())
    combined_stops = STOPWORDS | _STOPWORDS_EXTRA
    return [
        t for t in tokens
        if t not in combined_stops and len(t) >= 3
    ]


def _build_corpus(
    tokenised_docs: List[List[str]],
) -> Tuple[corpora.Dictionary, list]:
    """
    Build a Gensim Dictionary and BoW corpus from tokenised documents.
    Filters extremes to remove very rare and very common tokens.
    """
    dictionary = corpora.Dictionary(tokenised_docs)
    # Remove tokens that appear in fewer than 2 docs or more than 90 % of docs
    dictionary.filter_extremes(no_below=2, no_above=0.90)
    corpus = [dictionary.doc2bow(doc) for doc in tokenised_docs]
    return dictionary, corpus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_lda(
    sentences     : List[str],
    num_topics    : int   = 3,
    passes        : int   = 15,
    iterations    : int   = 100,
    random_state  : int   = 42,
    alpha         : str   = "auto",
    eta           : str   = "auto",
) -> LDAResult:
    """
    Train an LDA model on the provided sentences.

    Parameters
    ----------
    sentences    : List of pre-processed sentence strings.
    num_topics   : Number of latent topics K.
    passes       : Number of full corpus passes during training.
    iterations   : Max iterations per pass (E-step).
    random_state : Seed for reproducibility.
    alpha        : Document-topic prior ('auto', 'symmetric', or array).
    eta          : Topic-word prior  ('auto', 'symmetric', or array).

    Returns
    -------
    LDAResult named-tuple.
    """
    if not sentences:
        raise ValueError("sentences list is empty — cannot train LDA.")

    # --- Tokenise ---
    tokenised = [_tokenise(s) for s in sentences]
    non_empty = [(i, t) for i, t in enumerate(tokenised) if t]

    if len(non_empty) < num_topics:
        raise ValueError(
            f"Only {len(non_empty)} non-empty docs but num_topics={num_topics}. "
            "Reduce num_topics or provide more data."
        )

    # Keep only non-empty docs for training; map back to original index later
    train_indices  = [i for i, _ in non_empty]
    train_tokenised = [t for _, t in non_empty]

    # --- Corpus ---
    dictionary, corpus = _build_corpus(train_tokenised)

    if len(dictionary) == 0:
        raise ValueError(
            "Dictionary is empty after filtering. "
            "Try lowering filter_extremes thresholds."
        )

    # --- Train LDA ---
    lda_model = LdaModel(
        corpus        = corpus,
        id2word       = dictionary,
        num_topics    = num_topics,
        passes        = passes,
        iterations    = iterations,
        random_state  = random_state,
        alpha         = alpha,
        eta           = eta,
        per_word_topics = False,
    )

    # --- Infer topic distribution for EVERY sentence (including empties) ---
    all_doc_topics: List[List[Tuple[int, float]]] = []
    train_corpus_iter = iter(corpus)

    for i, tok in enumerate(tokenised):
        if i in train_indices:
            bow = next(train_corpus_iter)
            dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        else:
            # Empty sentence → uniform distribution
            dist = [(k, 1.0 / num_topics) for k in range(num_topics)]
        all_doc_topics.append(list(dist))

    return LDAResult(
        model      = lda_model,
        corpus     = corpus,
        dictionary = dictionary,
        doc_topics = all_doc_topics,
        num_topics = num_topics,
        sentences  = tokenised,
    )


def print_topics(lda_result: LDAResult, top_n: int = 10) -> None:
    """Pretty-print the top words for each discovered topic."""
    print(f"\n{'='*60}")
    print(f"LDA Topics  (K = {lda_result.num_topics})")
    print(f"{'='*60}")
    for topic_id in range(lda_result.num_topics):
        terms = lda_result.model.show_topic(topic_id, topn=top_n)
        words = ", ".join(f"{w}({p:.3f})" for w, p in terms)
        print(f"  Topic {topic_id}: {words}")
    print()
