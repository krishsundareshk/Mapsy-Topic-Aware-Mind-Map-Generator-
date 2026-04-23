"""
pipeline/evaluate_gt.py
========================
Ground-truth-based P/R/F1 evaluation for Phases 2 and 3.

Compares pipeline outputs against two ground-truth files:
  • gt_sentences.txt   — one topic block per ground-truth sentence group
  • gt_mindmap.json    — hierarchical noun-phrase ground truth (L1/L2/L3)

Metrics
-------
Phase 2 — Topic Segmentation
  • Per-topic and macro P/R/F1 (Hungarian-aligned by Jaccard sentence overlap)
  • TP = sentence in both GT topic and generated topic
  • FP = sentence in generated topic but wrong GT topic
  • FN = GT sentence missing from generated topic

Phase 3 — Concept Extraction
  • Flat P/R/F1  (all GT nodes across all L2 branches, per topic)
  • L2-level P/R/F1 (branch label match per topic)
  • L3-level P/R/F1 (leaf node match per aligned L2 branch)
  • Each at three tiers: Fuzzy, ROUGE-1 token, Semantic (BERTScore-style)

Usage
-----
    from pipeline.evaluate_gt import evaluate_gt, print_gt_report

    scores = evaluate_gt(
        sentence_groups   = cluster_result.sentence_groups,
        concept_results   = concept_results,
        hierarchies       = hierarchies,
        gt_sentences_path = "gt_sentences.txt",
        gt_mindmap_path   = "gt_mindmap.json",
    )
    print_gt_report(scores)
"""
from __future__ import annotations
import json, math, re
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from rapidfuzz import fuzz as _fuzz
from scipy.optimize import linear_sum_assignment

# ── optional sentence-transformer for semantic tier ──────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST, util as _util
    _ST_MODEL = _ST("all-MiniLM-L6-v2")
    _ST_OK = True
except ImportError:
    _ST_OK = False
    _ST_MODEL = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase, strip, remove trailing plural 's'."""
    return re.sub(r"\s+", " ", s.lower().strip()).rstrip("s")

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())

def _bow(labels: List[str]) -> Counter:
    return Counter(_tokens(" ".join(labels)))

def _fuzzy_prf(gen_labels: List[str], gt_labels: List[str],
               threshold: int = 80) -> Tuple[float, float, float]:
    """
    Fuzzy match using token_sort_ratio (handles word order differences too).
    threshold: 0-100. 80 means 80% character similarity required to count as a match.
    
    Examples at threshold=80:
      "electric" vs "electrical"          → ratio ~89  → MATCH
      "EV" vs "electric vehicle"          → ratio ~40  → NO MATCH
      "greenhouse gas" vs "greenhouse gases" → ratio ~96 → MATCH
      "solar energy" vs "solar panel"     → ratio ~75  → NO MATCH
    """
    if not gen_labels or not gt_labels:
        p = 0.0 if gen_labels else 1.0
        r = 0.0 if gt_labels  else 1.0
        f = 2 * p * r / (p + r + 1e-9)
        return round(p, 3), round(r, 3), round(f, 3)

    gen_norm = [_norm(g) for g in gen_labels]
    gt_norm  = [_norm(g) for g in gt_labels]

    # TP: GT phrases that have at least one Gen match above threshold
    tp = sum(
        1 for g in gt_norm
        if any(_fuzz.token_sort_ratio(g, p) >= threshold for p in gen_norm)
    )
    # FP: Gen phrases that match NO GT phrase
    fp = sum(
        1 for p in gen_norm
        if not any(_fuzz.token_sort_ratio(g, p) >= threshold for g in gt_norm)
    )
    fn = len(gt_norm) - tp

    prec = round(tp / (tp + fp + 1e-9), 3)
    rec  = round(tp / (tp + fn + 1e-9), 3)
    f1   = round(2 * prec * rec / (prec + rec + 1e-9), 3)
    return prec, rec, f1

def _rouge1_prf(gen_labels: List[str], gt_labels: List[str]) -> Tuple[float, float, float]:
    gen_bow = _bow(gen_labels);  gt_bow = _bow(gt_labels)
    overlap = sum((gen_bow & gt_bow).values())
    p = overlap / sum(gen_bow.values()) if gen_bow else 0.0
    r = overlap / sum(gt_bow.values())  if gt_bow  else 0.0
    f = 2 * p * r / (p + r + 1e-9)
    return round(p, 3), round(r, 3), round(f, 3)

def _semantic_prf(gen_labels: List[str], gt_labels: List[str]) -> Tuple[float, float, float]:
    """BERTScore-style: per-item best cosine match, then F1."""
    if not _ST_OK or not gen_labels or not gt_labels:
        return 0.0, 0.0, 0.0
    gen_embs = _ST_MODEL.encode(gen_labels, convert_to_tensor=True)
    gt_embs  = _ST_MODEL.encode(gt_labels,  convert_to_tensor=True)
    prec = float(_util.cos_sim(gen_embs, gt_embs).max(dim=1).values.mean())
    rec  = float(_util.cos_sim(gt_embs, gen_embs).max(dim=1).values.mean())
    f    = 2 * prec * rec / (prec + rec + 1e-9)
    return round(prec, 3), round(rec, 3), round(f, 3)

def _three_tier(gen_labels: List[str], gt_labels: List[str]) -> Dict[str, Any]:
    """Return fuzzy / rouge1 / semantic P, R, F1 for two label lists."""
    gen_norm = set(_norm(n) for n in gen_labels)
    gt_norm  = set(_norm(n) for n in gt_labels)
    fp, fr, ff = _fuzzy_prf(gen_labels, gt_labels, threshold=80)
    rp, rr, rf = _rouge1_prf(gen_labels, gt_labels)
    sp, sr, sf = _semantic_prf(gen_labels, gt_labels)
    return {
        "fuzzy":    {"P": fp, "R": fr, "F1": ff,
                     "matched": sorted(gen_norm & gt_norm)},
        "rouge1":   {"P": rp, "R": rr, "F1": rf},
        "semantic": {"P": sp, "R": sr, "F1": sf,
                     "note": "BERTScore-style; requires sentence-transformers"
                              if _ST_OK else "sentence-transformers not installed"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_sentences(path: str) -> Dict[int, List[str]]:
    """
    Load ground-truth sentence groups from a plain text file.

    Expected format (TOPIC blocks separated by blank lines):
        TOPIC 0
        Sentence one here.
        Sentence two here.

        TOPIC 1
        Another sentence.
        ...

    Returns {topic_id: [sentence_strings]}
    """
    groups: Dict[int, List[str]] = {}
    current_id: Optional[int] = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^TOPIC\s+(\d+)\s*$", line, re.IGNORECASE)
            if m:
                current_id = int(m.group(1))
                groups[current_id] = []
            elif current_id is not None and line:
                groups[current_id].append(line)
    return groups

def load_gt_mindmap(path: str) -> Dict[str, Any]:
    """Load gt_mindmap.json and return the parsed dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Topic Segmentation P/R/F1
# ─────────────────────────────────────────────────────────────────────────────

def _sent_jaccard(a: str, b: str) -> float:
    ta = set(_tokens(a)); tb = set(_tokens(b))
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0

def phase2_prf(
    gt_groups:  Dict[int, List[str]],
    gen_groups: Dict[int, List[str]],
) -> Dict[str, Any]:
    gt_ids  = sorted(gt_groups.keys())
    gen_ids = sorted(gen_groups.keys())
    K       = max(len(gt_ids), len(gen_ids))

    cost = np.zeros((K, K))
    for i, g in enumerate(gt_ids):
        for j, p in enumerate(gen_ids):
            a = gt_groups[g]; b = gen_groups[p]
            if not a or not b:
                cost[i][j] = 0.0
                continue
            score = sum(
                max(_sent_jaccard(gs, ps) for ps in b)
                for gs in a
            ) / len(a)
            cost[i][j] = -score
    row_ind, col_ind = linear_sum_assignment(cost)

    THRESH = 0.4
    per_topic: Dict[int, Dict] = {}

    for r, c in zip(row_ind, col_ind):
        if r >= len(gt_ids) or c >= len(gen_ids):
            continue
        gt_id    = gt_ids[r];  gen_id   = gen_ids[c]
        gt_list  = gt_groups[gt_id]
        gen_list = gen_groups[gen_id]

        tp = sum(1 for gs in gt_list
                 if any(_sent_jaccard(gs, ps) >= THRESH for ps in gen_list))
        fp = sum(1 for ps in gen_list
                 if not any(_sent_jaccard(gs, ps) >= THRESH for gs in gt_list))
        fn = len(gt_list) - tp

        p  = round(tp / (tp + fp + 1e-9), 3)
        rc = round(tp / (tp + fn + 1e-9), 3)
        f  = round(2 * p * rc / (p + rc + 1e-9), 3)

        per_topic[gt_id] = {
            "matched_gen_id": gen_id,
            "P": p, "R": rc, "F1": f,
            "TP": tp, "FP": fp, "FN": fn,
            "gt_size": len(gt_list),
            "gen_size": len(gen_list),
        }

    mp = sum(v["P"]  for v in per_topic.values()) / len(per_topic) if per_topic else 0.0
    mr = sum(v["R"]  for v in per_topic.values()) / len(per_topic) if per_topic else 0.0
    mf = sum(v["F1"] for v in per_topic.values()) / len(per_topic) if per_topic else 0.0

    return {
        "per_topic": per_topic,
        "macro": {"P": round(mp, 3), "R": round(mr, 3), "F1": round(mf, 3)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Concept Extraction P/R/F1 (three levels × three tiers)
# ─────────────────────────────────────────────────────────────────────────────

def _align_l2(gt_l2_nodes: List[Dict], gen_l2_labels: List[str]) -> Dict[str, str]:
    """
    Align GT L2 branch labels to generated L2 branch labels via Hungarian
    on normalised token overlap.
    Returns {gt_label: best_gen_label}.
    """
    gt_labels = [n["label"] for n in gt_l2_nodes]
    K = max(len(gt_labels), len(gen_l2_labels))
    cost = np.zeros((K, K))
    for i, g in enumerate(gt_labels):
        for j, p in enumerate(gen_l2_labels):
            gn = set(_tokens(g));  pn = set(_tokens(p))
            cost[i][j] = -(len(gn & pn) / len(gn | pn)) if (gn | pn) else 0.0
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: Dict[str, str] = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(gt_labels) and c < len(gen_l2_labels):
            mapping[gt_labels[r]] = gen_l2_labels[c]
    return mapping


def phase3_prf(
    gt_mindmap:      Dict[str, Any],
    concept_results: List[Dict[str, Any]],
    hierarchies:     List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Three-level concept P/R/F1 evaluation.

    concept_results : Phase 3 output (graph nodes per topic)
    hierarchies     : Phase 4 output (tree structure per topic)
    gt_mindmap      : Loaded gt_mindmap.json dict

    Returns per-topic scores at three levels × three tiers + macro averages.
    """
    gt_by_id = {t["id"]: t for t in gt_mindmap["topics"]}

    gen_nodes_by_tid: Dict[int, List[str]] = {}
    for cr in concept_results:
        tid = cr.get("topic_id", -1)
        gen_nodes_by_tid[tid] = [
            info["label"] for info in cr["graph"]["nodes"].values()
        ]

    hier_by_tid: Dict[int, Dict] = {}
    for h in hierarchies:
        tid = h.get("topic_id", -1)
        hier_by_tid[tid] = h

    results: Dict[int, Dict] = {}

    for gt_tid, gt_topic in gt_by_id.items():
        gt_l1_label = gt_topic["label"]
        gt_l2_nodes = gt_topic["l2_nodes"]

        # ── Flat L3: all GT leaf nodes vs all generated nodes ────────────────
        gt_all_l3 = [n for l2 in gt_l2_nodes for n in l2["l3_nodes"]]
        gen_all   = gen_nodes_by_tid.get(gt_tid, [])
        flat_l3   = _three_tier(gen_all, gt_all_l3)

        # ── L2 branch label match ─────────────────────────────────────────────
        gt_l2_labels  = [l2["label"] for l2 in gt_l2_nodes]
        gen_hier       = hier_by_tid.get(gt_tid, {})
        gen_l2_labels  = [c.get("label", "") for c in gen_hier.get("children", [])]
        l2_scores      = _three_tier(gen_l2_labels, gt_l2_labels)

        # ── L3 per aligned L2 branch ──────────────────────────────────────────
        l2_alignment = _align_l2(gt_l2_nodes, gen_l2_labels)
        gen_l2_map   = {c.get("label", ""): [gc.get("label", "")
                         for gc in c.get("children", [])]
                        for c in gen_hier.get("children", [])}

        per_l2: Dict[str, Dict] = {}
        for gt_l2_node in gt_l2_nodes:
            gt_l2_lbl   = gt_l2_node["label"]
            gt_l3_list  = gt_l2_node["l3_nodes"]
            gen_l2_lbl  = l2_alignment.get(gt_l2_lbl, "")
            gen_l3_list = gen_l2_map.get(gen_l2_lbl, [])
            per_l2[gt_l2_lbl] = {
                "matched_gen_l2": gen_l2_lbl,
                "gt_l3_count":    len(gt_l3_list),
                "gen_l3_count":   len(gen_l3_list),
                **_three_tier(gen_l3_list, gt_l3_list),
            }

        def _macro_tier(per_l2_dict, tier):
            vals = [v[tier] for v in per_l2_dict.values() if tier in v]
            if not vals: return {"P": 0.0, "R": 0.0, "F1": 0.0}
            return {k: round(sum(v[k] for v in vals)/len(vals), 3) for k in ["P","R","F1"]}

        results[gt_tid] = {
            "l1_label":      gt_l1_label,
            "flat_l3":       flat_l3,
            "l2_match":      l2_scores,
            "per_l2_branch": per_l2,
            "l3_macro": {
                tier: _macro_tier(per_l2, tier)
                for tier in ["fuzzy", "rouge1", "semantic"]
            },
        }

    def _global_macro(level_key, tier):
        vals = [results[tid][level_key][tier] for tid in results if level_key in results[tid]]
        if not vals: return {"P": 0.0, "R": 0.0, "F1": 0.0}
        return {k: round(sum(v[k] for v in vals)/len(vals), 3) for k in ["P","R","F1"]}

    global_macro = {
        "flat_l3":  {tier: _global_macro("flat_l3",  tier) for tier in ["fuzzy","rouge1","semantic"]},
        "l2_match": {tier: _global_macro("l2_match", tier) for tier in ["fuzzy","rouge1","semantic"]},
        "l3_macro": {tier: {k: round(sum(
                        results[tid]["l3_macro"][tier][k] for tid in results
                     ) / max(len(results), 1), 3) for k in ["P","R","F1"]}
                     for tier in ["fuzzy","rouge1","semantic"]},
    }

    return {"per_topic": results, "global_macro": global_macro}


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gt(
    sentence_groups:   Dict[int, List[str]],
    concept_results:   List[Dict[str, Any]],
    hierarchies:       List[Dict[str, Any]],
    gt_sentences_path: str = "gt_sentences.txt",
    gt_mindmap_path:   str = "gt_mindmap.json",
) -> Dict[str, Any]:
    gt_sentences = load_gt_sentences(gt_sentences_path)
    gt_mindmap   = load_gt_mindmap(gt_mindmap_path)

    print("  [GT Eval] Phase 2 — Topic Segmentation P/R/F1 ...")
    p2 = phase2_prf(gt_sentences, sentence_groups)

    print("  [GT Eval] Phase 3 — Concept Extraction P/R/F1 (3 levels × 3 tiers) ...")
    p3 = phase3_prf(gt_mindmap, concept_results, hierarchies)

    return {"phase2": p2, "phase3": p3}


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable report
# ─────────────────────────────────────────────────────────────────────────────

def print_gt_report(scores: Dict[str, Any]) -> None:
    W = 65
    print("\n" + "═" * W)
    print("  GT-Based Evaluation Report  (P/R/F1 vs Ground Truth)")
    print("═" * W)

    p2 = scores.get("phase2", {})
    print("\n  PHASE 2 · Topic Segmentation")
    print("  " + "─" * (W - 2))
    for gt_id, v in sorted(p2.get("per_topic", {}).items()):
        print(f"    GT Topic {gt_id} ↔ Gen Topic {v['matched_gen_id']}  "
              f"P={v['P']}  R={v['R']}  F1={v['F1']}  "
              f"(TP={v['TP']} FP={v['FP']} FN={v['FN']})")
    m = p2.get("macro", {})
    print(f"\n    Macro avg  P={m.get('P',0)}  R={m.get('R',0)}  F1={m.get('F1',0)}")

    p3 = scores.get("phase3", {})
    tiers = ["fuzzy", "rouge1", "semantic"]
    tier_names = {"fuzzy": "Fuzzy Match (>=80%)", "rouge1": "ROUGE-1 Token",
                  "semantic": "Semantic (BERTScore)"}

    for tid, tv in sorted(p3.get("per_topic", {}).items()):
        print(f"\n  PHASE 3 · Topic {tid}: {tv['l1_label']}")
        print("  " + "─" * (W - 2))

        print("    [Flat L3 — all leaf nodes]")
        for tier in tiers:
            v = tv["flat_l3"][tier]
            print(f"      {tier_names[tier]:<28} P={v['P']}  R={v['R']}  F1={v['F1']}")

        print("    [L2 Branch Labels]")
        for tier in tiers:
            v = tv["l2_match"][tier]
            print(f"      {tier_names[tier]:<28} P={v['P']}  R={v['R']}  F1={v['F1']}")

        print("    [L3 per aligned L2 branch]")
        for gt_l2_lbl, bv in tv.get("per_l2_branch", {}).items():
            print(f"      Branch: GT='{gt_l2_lbl}' ↔ Gen='{bv['matched_gen_l2']}'  "
                  f"(GT L3={bv['gt_l3_count']} Gen L3={bv['gen_l3_count']})")
            for tier in tiers:
                v = bv[tier]
                print(f"        {tier_names[tier]:<28} P={v['P']}  R={v['R']}  F1={v['F1']}")

        print("    [L3 macro across branches]")
        for tier in tiers:
            v = tv["l3_macro"][tier]
            print(f"      {tier_names[tier]:<28} P={v['P']}  R={v['R']}  F1={v['F1']}")

    gm = p3.get("global_macro", {})
    print(f"\n  PHASE 3 · Global Macro (all topics)")
    print("  " + "─" * (W - 2))
    for level, level_name in [("flat_l3","Flat L3"),("l2_match","L2 Branch"),
                               ("l3_macro","L3 per-branch macro")]:
        print(f"    [{level_name}]")
        for tier in tiers:
            v = gm.get(level, {}).get(tier, {})
            print(f"      {tier_names[tier]:<28} P={v.get('P',0)}  R={v.get('R',0)}  F1={v.get('F1',0)}")

    print("\n" + "═" * W + "\n")