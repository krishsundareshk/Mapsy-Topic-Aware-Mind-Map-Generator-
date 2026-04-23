"""
pipeline/phase2_thematic_clustering_lda.py — original LDA backend
Used when main.py is run with --topic-model lda
"""
from __future__ import annotations

import os, sys
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.thematic_clustering_lda import (
    run_thematic_clustering,
    ThematicConfig,
    ThematicClusteringResult,
)


def run(
    sentences : List[str],
    k_min     : int  = 2,
    k_max     : int  = 5,
    verbose   : bool = False,
    **kwargs,                  # absorbs extra kwargs from main.py safely
) -> ThematicClusteringResult:
    cfg = ThematicConfig(
        k_min   = k_min,
        k_max   = k_max,
        verbose = verbose,
    )
    return run_thematic_clustering(sentences, cfg)