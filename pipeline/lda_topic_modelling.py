"""
pipeline/lda_topic_modelling.py — shim
=======================================
LDA backend replaced by BERTopic. All existing imports of LDAResult,
run_lda, and print_topics continue to work unchanged — they now resolve
to their BERTopic equivalents transparently.
"""
from pipeline.bertopic_modelling import (   # noqa: F401
    BERTopicResult as LDAResult,
    BERTopicResult,
    _ModelWrapper,
    run_lda,
    print_topics,
    compute_embeddings,
)

__all__ = [
    "LDAResult",
    "BERTopicResult",
    "_ModelWrapper",
    "run_lda",
    "print_topics",
    "compute_embeddings",
]