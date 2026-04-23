"""
pipeline/phase4_kmeans.py
=========================
Phase 4 — Hierarchy Construction  |  Clustering: K-Means++

Algorithm
---------
L1 : Root label from Phase 3 degree-central root_node (LDA fallback).
L2 : K-Means++ on TF-IDF vectors of concept node labels.
     K chosen by _choose_k() step function based on node count.
     Tiny clusters (< 2 members) absorbed into nearest centroid.
L3 : SPO-edge traversal from L2 representative + sibling fill-in (cap 8).
"""
from __future__ import annotations
import re, math, random
from collections import defaultdict, Counter
from typing import List, Dict, Any, Set, Optional

# ── WEF ──────────────────────────────────────────────────────────────────────
def _wef(graph):
    nodes = graph.get("nodes", {}); edges = graph.get("edges", [])
    if not nodes: return {}
    degree = defaultdict(int)
    for e in edges:
        for k in ("source", "target"):
            if e.get(k): degree[e[k]] += 1
    md = max(degree.values(), default=1) or 1
    mf = max(v["freq"] for v in nodes.values()) or 1
    ml = max(len(v["label"].split()) for v in nodes.values()) or 1
    return {
        nid: (0.3 * len(i["label"].split()) / ml
            + 0.4 * degree.get(nid, 0) / md
            + 0.3 * i["freq"] / mf)
        for nid, i in nodes.items()
    }

# ── Label helpers ─────────────────────────────────────────────────────────────
_FUNC = {
    "the","a","an","of","in","on","at","by","for","and","or","to",
    "with","as","is","are","was","were","be","been","has","have",
    "had","not","more","most","very","also","even","its","this",
    "that","these","those","each","every","any","all","both",
}
_GENERIC_LDA = {
    "energy","power","use","work","make","form","part","type","level","rate","area",
    "data","way","time","year","world","life","day","thing","place","point","result",
    "process","system","method","also","would","could","should","may","might","must",
    "can","will","even","still","already","used","one","two","new","many","much","well",
    "first","second","next","last","high","low","large","small","good","bad","great",
    "major","like","change","include","provide","increase","number","amount","global",
    "local","general","specific","common","based","related","given","known","called",
    "such","other","same","different","various","several","certain","following",
    "important","significant","critical","key","main","sources","source",
}

def _clean(label):
    label = re.sub(r"\s+", " ", re.sub(r"-", " ", label)).strip()
    words = label.split()
    while words and words[0].lower()  in _FUNC: words = words[1:]
    while words and words[-1].lower() in _FUNC: words = words[:-1]
    return " ".join(w.capitalize() for w in words) if words else label.title()

def _lda_label(lda_result, tid):
    words = [w for w, _ in lda_result.model.show_topic(tid, topn=15)
             if w not in _GENERIC_LDA and len(w) > 3]
    return " ".join(w.capitalize() for w in words[:2]) if words else f"Topic {tid}"

def _root_label(graph, lda_result, tid):
    rn = graph.get("root_node"); nodes = graph.get("nodes", {})
    if rn and rn in nodes: return _clean(nodes[rn]["label"])
    try:    return _lda_label(lda_result, tid)
    except: return f"Topic {tid}"

# ── TF-IDF vectors ─────────────────────────────────────────────────────────────
def _tfidf_vecs(labels):
    n = len(labels)
    tokenised = [lbl.lower().split() for lbl in labels]
    df = Counter(w for tok in tokenised for w in set(tok))
    vocab = sorted(df); vidx = {w: i for i, w in enumerate(vocab)}
    dim = len(vocab) or 1
    vecs = []
    for tok in tokenised:
        tf = Counter(tok); total = len(tok) or 1
        v = [0.0] * dim
        for w, c in tf.items():
            if w in vidx:
                v[vidx[w]] = (c / total) * (math.log((1 + n) / (1 + df[w])) + 1)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs

def _choose_k(n):
    return 1 if n <= 2 else 2 if n <= 5 else 3 if n <= 10 else 4 if n <= 16 else 5

def _d2(a, b): return sum((x - y) ** 2 for x, y in zip(a, b))

def _kmeans_pp(ids, vecs, k, seed=42):
    n = len(ids)
    if not n: return {}
    k = min(k, n)
    if k == 1: return {nid: 0 for nid in ids}
    rng = random.Random(seed); dim = len(vecs[0])
    centres = [list(vecs[rng.randint(0, n - 1)])]
    for _ in range(k - 1):
        dists = [min(_d2(v, c) for c in centres) for v in vecs]
        tot = sum(dists) or 1.0; r, cum, chosen = rng.random(), 0.0, n - 1
        for i, p in enumerate(dists):
            cum += p / tot
            if r <= cum: chosen = i; break
        centres.append(list(vecs[chosen]))
    assign = [0] * n
    for _ in range(50):
        new_a = [min(range(k), key=lambda c, i=i: _d2(vecs[i], centres[c])) for i in range(n)]
        if new_a == assign: break
        assign = new_a
        sums = [[0.0] * dim for _ in range(k)]; cnts = [0] * k
        for i, c in enumerate(assign):
            cnts[c] += 1
            for d in range(dim): sums[c][d] += vecs[i][d]
        centres = [[sums[c][d] / cnts[c] if cnts[c] else 0.0 for d in range(dim)] for c in range(k)]
    cl = defaultdict(list)
    for i, nid in enumerate(ids): cl[assign[i]].append(nid)
    # Absorb tiny clusters into nearest centroid
    changed = True
    while changed:
        changed = False
        for cid in list(cl):
            if len(cl[cid]) < 2 and len(cl) > 1:
                members = cl.pop(cid)
                idx_map = {nid: ids.index(nid) for nid in members}
                c_ctr = [sum(vecs[idx_map[m]][d] for m in members) / len(members) for d in range(dim)]
                best = min(cl, key=lambda c: _d2(
                    c_ctr,
                    [sum(vecs[ids.index(m)][d] for m in cl[c]) / len(cl[c]) for d in range(dim)]
                ))
                cl[best].extend(members); changed = True; break
    out = {}
    for new_cid, (_, members) in enumerate(cl.items()):
        for nid in members: out[nid] = new_cid
    return out

# ── Adjacency + L3 ─────────────────────────────────────────────────────────────
def _build_adj(edges, root_nid):
    adj = defaultdict(list)
    for e in edges:
        s, t, p = e.get("source",""), e.get("target",""), e.get("predicate","related to")
        if not s or not t: continue
        if s != root_nid: adj[s].append((t, p, "out"))
        if t != root_nid: adj[t].append((s, p, "in"))
    return adj

def _l3_from_edges(l2_nid, adj, nodes, wef, used, max_l3=6):
    nb = sorted(adj.get(l2_nid, []), key=lambda x: (0 if x[2]=="out" else 1, -wef.get(x[0], 0)))
    seen, ch = set(), []
    for nb_nid, pred, dir_ in nb:
        if nb_nid in used or nb_nid in seen or nb_nid not in nodes: continue
        seen.add(nb_nid)
        ch.append({"id": nb_nid, "label": _clean(nodes[nb_nid]["label"]),
                   "predicate": pred, "direction": dir_,
                   "wef_score": round(wef.get(nb_nid, 0), 4)})
        if len(ch) >= max_l3: break
    if not ch:
        for nid in sorted(nodes, key=lambda n: -wef.get(n, 0)):
            if nid not in used and nid != l2_nid and nid not in seen:
                ch.append({"id": nid, "label": _clean(nodes[nid]["label"]),
                           "predicate": "related to", "direction": "out",
                           "wef_score": round(wef.get(nid, 0), 4)})
                if len(ch) >= max_l3: break
    return ch

# ── Build one hierarchy ────────────────────────────────────────────────────────
def _build_one(topic_id, concept_result, lda_result, depth):
    graph = concept_result.get("graph", {"nodes": {}, "edges": []})
    nodes = graph.get("nodes", {}); edges = graph.get("edges", [])
    rn    = graph.get("root_node")
    rl    = _root_label(graph, lda_result, topic_id)
    wef   = _wef(graph)
    sids  = [nid for nid in sorted(wef, key=wef.get, reverse=True) if nid != rn]
    rw    = wef.get(rn, 0.0) if rn else (sum(wef.values()) / len(wef) if wef else 0.0)

    if depth == 1 or not sids:
        return {"topic_id": topic_id, "label": rl, "wef_score": round(rw, 4), "children": []}

    vecs  = _tfidf_vecs([nodes[nid]["label"] for nid in sids])
    k     = _choose_k(len(sids))
    cmap  = _kmeans_pp(sids, vecs, k)
    clusters = defaultdict(list)
    for nid, cid in cmap.items(): clusters[cid].append(nid)
    for cid in clusters: clusters[cid].sort(key=lambda n: -wef.get(n, 0))

    max_l2 = 5 if depth == 3 else 4
    while len(clusters) > max_l2:
        sc = sorted(clusters, key=lambda c: len(clusters[c]))
        clusters[sc[1]].extend(clusters.pop(sc[0]))

    adj = _build_adj(edges, rn); used_l2: Set[str] = set(); l2_nodes = []
    for cid in sorted(clusters, key=lambda c: -sum(wef.get(n, 0) for n in clusters[c])):
        members = clusters[cid]; best = members[0]
        l2_label = _clean(nodes[best]["label"])
        l2_wef   = sum(wef.get(n, 0) for n in members) / len(members)
        used_l2.add(best)
        if depth == 2:
            l2_nodes.append({"id": f"l2_{topic_id}_{cid}", "label": l2_label,
                              "wef_score": round(l2_wef, 4), "children": []})
        else:
            l3ch = _l3_from_edges(best, adj, nodes, wef, used_l2, 6)
            for sib in members[1:]:
                if len(l3ch) >= 8: break
                if sib in used_l2 or any(c["id"] == sib for c in l3ch): continue
                l3ch.append({"id": sib, "label": _clean(nodes[sib]["label"]),
                             "predicate": "related to", "direction": "out",
                             "wef_score": round(wef.get(sib, 0), 4)})
            l2_nodes.append({"id": f"l2_{topic_id}_{cid}", "label": l2_label,
                              "wef_score": round(l2_wef, 4), "children": l3ch})
    return {"topic_id": topic_id, "label": rl, "wef_score": round(rw, 4), "children": l2_nodes}

# ── Public API ─────────────────────────────────────────────────────────────────
def run(sentence_groups, concept_results, lda_result, depth=3, verbose=False):
    cr_by_id = {r["topic_id"]: r for r in concept_results}
    out = []
    for tid in sorted(sentence_groups):
        cr = cr_by_id.get(tid, {"graph": {"nodes": {}, "edges": []}, "sentences": []})
        h  = _build_one(tid, cr, lda_result, depth)
        if verbose:
            print(f"  [Phase4-KMeans++] Topic {tid} root='{h['label']}': "
                  f"L2={len(h['children'])}  "
                  f"L3={sum(len(c['children']) for c in h['children'])}  "
                  f"WEF={h['wef_score']:.3f}")
        out.append(h)
    return out