"""
main.py — Topic-Aware Mind Map Generator
==========================================
Runs all 5 phases end-to-end. Depth is selectable both here (sets
the default) and interactively inside the browser (no page reload).

Usage
-----
  python main.py                                               # paste text interactively
  python main.py --input document.txt                         # raw text file
  python main.py --input sentences.txt --clean                # pre-split sentences
  python main.py --input doc.txt --depth 2                    # default depth (1/2/3)
  python main.py --input doc.txt --k-min 2 --k-max 6          # topic count range
  python main.py --input doc.txt --output report.html         # custom output path
  python main.py --input doc.txt --clusterer birch            # BIRCH for Phase 4
  python main.py --input doc.txt --clusterer dbscan           # DBSCAN for Phase 4
  python main.py --input doc.txt --extractor bert             # BERT for Phase 3
  python main.py --input doc.txt --extractor spo              # rule-based SPO for Phase 3
  python main.py --input doc.txt --topic-model bertopic       # BERTopic backend (default)
  python main.py --input doc.txt --topic-model lda            # original LDA backend
  python main.py --input doc.txt --quiet                      # suppress verbose output
"""
from __future__ import annotations
import argparse, os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'pipeline'))

from pipeline.phase1_preprocessing import preprocess
from pipeline.phase5_visualisation  import run as render


def _hdr(n: int, title: str, verbose: bool) -> None:
    if verbose:
        print(f"\n{'─'*65}\n  Phase {n}: {title}\n{'─'*65}")


def _t(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def _load_phase2(topic_model: str):
    if topic_model == 'lda':
        from pipeline.phase2_thematic_clustering_lda import run as run_p2
    else:
        from pipeline.phase2_thematic_clustering import run as run_p2
    return run_p2


def _load_phase3(extractor: str):
    if extractor == 'bert':
        from pipeline.phase3_bert import run as run_p3
    else:
        from pipeline.phase3_concept_extraction import run as run_p3
    return run_p3


def _load_phase4(clusterer: str):
    if clusterer == 'birch':
        from pipeline.phase4_birch  import run as run_p4
    elif clusterer == 'dbscan':
        from pipeline.phase4_dbscan import run as run_p4
    else:
        from pipeline.phase4_hierarchy import run as run_p4
    return run_p4


def run_pipeline(
    raw_text    : str  = '',
    sentences   : list = None,
    depth       : int  = 3,
    k_min       : int  = 2,
    k_max       : int  = 5,
    output_path : str  = 'mindmap.html',
    clusterer   : str  = 'kmeans',
    extractor   : str  = 'spo',
    topic_model : str  = 'bertopic',
    verbose     : bool = True,
) -> str:
    t_all = time.time()

    if verbose:
        print("\n" + "═"*65)
        print("  🧠  Topic-Aware Mind Map Generator")
        print(f"  📊  Phase 2 topic model : {topic_model.upper()}")
        print(f"  🔬  Phase 3 extractor   : {extractor.upper()}")
        print(f"  🔧  Phase 4 clusterer   : {clusterer.upper()}")
        print("═"*65)

    # ── Phase 1: Pre-Processing ───────────────────────────────────────────────
    _hdr(1, "Pre-Processing", verbose)
    t0 = time.time()
    if sentences is None:
        sentences = preprocess(raw_text, verbose=verbose)
    if not sentences:
        raise ValueError("No sentences produced — check input text.")
    if verbose:
        print(f"  ✓ {len(sentences)} clean sentences  ({_t(t0)})")

    # ── Phase 2: Thematic Clustering ──────────────────────────────────────────
    backend_label = (
        "BERTopic + Meta-Clustering + UMass coherence"
        if topic_model == 'bertopic'
        else "LDA + Coherence + TF-IDF + Dominance"
    )
    _hdr(2, f"Thematic Clustering  ({backend_label})", verbose)
    t0     = time.time()
    run_p2 = _load_phase2(topic_model)
    cluster_result  = run_p2(sentences, k_min=k_min, k_max=k_max, verbose=verbose)
    sentence_groups = cluster_result.sentence_groups
    lda_result      = cluster_result.lda_result

    if verbose:
        print(f"  ✓ K*={cluster_result.optimal_k} topics  ({_t(t0)})")
        for tid, sents in sentence_groups.items():
            print(f"    Topic {tid}: {len(sents)} sentences")

    # ── Phase 3: Concept Extraction ───────────────────────────────────────────
    p3_label = (
        "BERT semantic scoring + BERT predicates"
        if extractor == 'bert'
        else "noun chunks + TF-IDF + SPO co-occurrence"
    )
    _hdr(3, f"Concept Extraction  ({p3_label})", verbose)
    t0     = time.time()
    run_p3 = _load_phase3(extractor)
    concept_results = run_p3(sentence_groups, verbose=verbose)
    if verbose:
        nodes = sum(len(r['graph']['nodes']) for r in concept_results)
        edges = sum(len(r['graph']['edges']) for r in concept_results)
        print(f"  ✓ {nodes} concept nodes, {edges} edges  ({_t(t0)})")

    # ── Phase 4: Hierarchy Construction ───────────────────────────────────────
    _hdr(4, f"Hierarchy Construction  ({clusterer.upper()} clustering + WEF scoring)", verbose)
    t0     = time.time()
    run_p4 = _load_phase4(clusterer)
    hierarchies = run_p4(
        sentence_groups = sentence_groups,
        concept_results = concept_results,
        lda_result      = lda_result,
        depth           = depth,
        verbose         = verbose,
    )
    if verbose:
        print(f"  ✓ {len(hierarchies)} hierarchies  ({_t(t0)})")

    # ── Phase 5: Visualisation ────────────────────────────────────────────────
    _hdr(5, "Visualisation  (interactive HTML — depth-switchable in browser)", verbose)
    t0 = time.time()
    out = render(
        hierarchies = hierarchies,
        output_path = output_path,
        title       = f"Mind Map  [{topic_model.upper()} | {extractor.upper()} | {clusterer.upper()}]",
        verbose     = verbose,
    )
    if verbose:
        print(f"  ✓ HTML rendered  ({_t(t0)})")
        print(f"\n{'═'*65}")
        print(f"  ✅  Total time   : {_t(t_all)}")
        print(f"  📊  Topic model  : {topic_model.upper()}")
        print(f"  🔬  Extractor    : {extractor.upper()}")
        print(f"  🔧  Clusterer    : {clusterer.upper()}")
        print(f"  📄  Output       : {os.path.abspath(out)}")
        print(f"{'═'*65}\n")

    # ── GT-based P/R/F1 evaluation (optional — runs only if GT files exist) ──────
    from pipeline.evaluate_gt import evaluate_gt, print_gt_report
    _GT_SENTS = os.path.join(_HERE, "gt_sentences.txt")
    _GT_MAP   = os.path.join(_HERE, "gt_mindmap.json")
    if os.path.isfile(_GT_SENTS) and os.path.isfile(_GT_MAP):
        gt_scores = evaluate_gt(
            sentence_groups   = sentence_groups,
            concept_results   = concept_results,
            hierarchies       = hierarchies,
            gt_sentences_path = _GT_SENTS,
            gt_mindmap_path   = _GT_MAP,
        )
        print_gt_report(gt_scores)

    return os.path.abspath(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Topic-Aware Mind Map Generator — 5-phase NLP pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--input',       '-i', default=None,
                    help='Path to raw text file (or clean sentences with --clean)')
    ap.add_argument('--clean',       '-c', action='store_true',
                    help='Input already has one clean sentence per line')
    ap.add_argument('--output',      '-o', default='mindmap.html',
                    help='Output HTML file (default: mindmap.html)')
    ap.add_argument('--depth',       '-d', type=int, choices=[1, 2, 3], default=3,
                    help='Default depth: 1=Overview  2=Standard  3=Detailed')
    ap.add_argument('--k-min',       type=int, default=2,
                    help='Min topics for Phase 2 (default: 2)')
    ap.add_argument('--k-max',       type=int, default=5,
                    help='Max topics for Phase 2 (default: 5)')
    ap.add_argument('--clusterer',   choices=['kmeans', 'birch', 'dbscan'], default='kmeans',
                    help='Phase 4 clustering algorithm (default: kmeans)')
    ap.add_argument('--extractor',   choices=['spo', 'bert'], default='spo',
                    help='Phase 3 concept extractor (default: spo)')
    ap.add_argument('--topic-model', choices=['bertopic', 'lda'], default='bertopic',
                    dest='topic_model',
                    help='Phase 2 backend: bertopic (default) or lda')
    ap.add_argument('--quiet',       '-q', action='store_true',
                    help='Suppress verbose pipeline output')
    args = ap.parse_args()

    verbose = not args.quiet

    # ── Acquire input ─────────────────────────────────────────────────────────
    if args.input:
        if not os.path.isfile(args.input):
            print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()
        if args.clean:
            sentences = [ln.strip() for ln in content.splitlines() if ln.strip()]
            raw_text  = ''
        else:
            raw_text, sentences = content, None
    else:
        print("\n📝  Paste your text below.")
        print("    Type  END  on its own line when done (or press Ctrl+D):\n")
        lines: list = []
        try:
            while True:
                line = input()
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
        except EOFError:
            pass
        raw_text, sentences = '\n'.join(lines), None

    if not raw_text and not sentences:
        print("[ERROR] No input provided.", file=sys.stderr)
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        out = run_pipeline(
            raw_text    = raw_text,
            sentences   = sentences,
            depth       = args.depth,
            k_min       = args.k_min,
            k_max       = args.k_max,
            output_path = args.output,
            clusterer   = args.clusterer,
            extractor   = args.extractor,
            topic_model = args.topic_model,
            verbose     = verbose,
        )
        print(f"✅  Mind map saved: {out}")
        print("    Open in any browser.")
        print("    Use the  L1 / L2 / L3  buttons in the header to switch depth.\n")
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()