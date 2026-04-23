"""
pipeline/phase2_thematic_clustering.py — BERTopic backend
"""
from __future__ import annotations

import os, sys
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.thematic_clustering import (
    run_thematic_clustering,
    ThematicConfig,
    ThematicClusteringResult,
)


def run(
    sentences          : List[str],
    k_min              : int  = 2,
    k_max              : int  = 5,
    verbose            : bool = False,
    use_meta_clustering: bool = True,
    meta_k_max         : int  = 5,
) -> ThematicClusteringResult:
    cfg = ThematicConfig(
        k_min               = k_min,
        k_max               = k_max,
        random_state        = 42,
        k_strategy          = "max_coherence",
        top_n_words         = 12,
        coherence_top_n     = 10,
        lda_weight          = 0.4,
        cosine_weight       = 0.6,
        dominance_threshold = None,
        auto_percentile     = 0.15,
        use_fallback        = True,
        embedding_model     = "all-MiniLM-L6-v2",
        verbose             = verbose,
        use_meta_clustering = use_meta_clustering,
        meta_k_max          = meta_k_max,
    )
    return run_thematic_clustering(sentences, cfg)