"""
pipeline/phase4_hierarchy.py — v3
====================================
Changes from v2
---------------
1. Max L3 raised to 6 (was 5); sibling fill-in cap raised to 8 so
   dense topics don't leave L2 nodes childless.

2. wef_score is now included on every output node (L1, L2, L3) so
   Phase 5 can use it for dynamic sizing in future iterations.

3. Everything else (head-noun clustering, WEF formula, SPO-driven L3
   children, LDA fallback label) is identical to v2.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# WEF scorer
# ─────────────────────────────────────────────────────────────────────────────

def _wef(graph: Dict[str, Any]) -> Dict[str, float]:
    nodes = graph.get('nodes', {})
    edges = graph.get('edges', [])
    if not nodes: return {}
    degree: Dict[str,int] = defaultdict(int)
    for e in edges:
        for k in ('source','target'):
            if e.get(k): degree[e[k]] += 1
    max_deg  = max(degree.values(), default=1) or 1
    max_freq = max(v['freq'] for v in nodes.values()) or 1
    max_wlen = max(len(v['label'].split()) for v in nodes.values()) or 1
    return {
        nid: (
            0.3 * len(info['label'].split()) / max_wlen
          + 0.4 * degree.get(nid, 0) / max_deg
          + 0.3 * info['freq'] / max_freq
        )
        for nid, info in nodes.items()
    }

# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

_FUNC = {
    'the','a','an','of','in','on','at','by','for','and','or','to',
    'with','as','is','are','was','were','be','been','has','have',
    'had','not','more','most','very','also','even','its','this',
    'that','these','those','each','every','any','all','both',
}

def _clean(label: str) -> str:
    label = re.sub(r'\s+', ' ', label)
    label = re.sub(r'-', ' ', label).strip()
    words = label.split()
    while words and words[0].lower()  in _FUNC: words = words[1:]
    while words and words[-1].lower() in _FUNC: words = words[:-1]
    if not words: return label.title()
    return ' '.join(w.capitalize() for w in words)

def _head_noun(label: str) -> str:
    words = [w for w in label.lower().split() if w not in _FUNC]
    return words[-1] if words else label.lower()

_GENERIC_LDA = {
    'energy','power','use','work','make','form','part','type','level',
    'rate','area','data','way','time','year','world','life','day','thing',
    'place','point','result','process','system','method','approach',
    'issue','aspect','factor','condition','situation','context',
    'version','also','would','could','should','may','might','must',
    'can','will','even','still','around','across','within','every',
    'each','already','used','one','two','new','many','much','well',
    'first','second','next','last','high','low','large','small','good',
    'bad','great','major','like','change','today','include','including',
    'provide','provides','increase','increases','using','making',
    'number','amount','global','local','general','specific','common',
    'based','related','given','known','called','such','other','same',
    'different','various','several','certain','following',
    'important','significant','critical','key','main',
    'sources','source','forms','types','kinds','methods','ways',
}

def _lda_label(lda_result, topic_id: int) -> str:
    words = [w for w, _ in lda_result.model.show_topic(topic_id, topn=15)
             if w not in _GENERIC_LDA and len(w) > 3]
    if not words: return f'Topic {topic_id}'
    top = [w.capitalize() for w in words[:2]]
    return ' '.join(top)

def _root_label(graph: Dict[str, Any], lda_result, topic_id: int) -> str:
    root_nid = graph.get('root_node')
    nodes    = graph.get('nodes', {})
    if root_nid and root_nid in nodes:
        return _clean(nodes[root_nid]['label'])
    try:
        return _lda_label(lda_result, topic_id)
    except Exception:
        return f'Topic {topic_id}'

# ─────────────────────────────────────────────────────────────────────────────
# Head-noun clustering  (L2 groups)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_by_head(
    node_ids:   List[str],
    nodes:      Dict[str, Dict],
    wef_scores: Dict[str, float],
    max_l2:     int = 5,
    min_size:   int = 1,
) -> Dict[str, List[str]]:
    head_groups: Dict[str, List[str]] = defaultdict(list)
    for nid in node_ids:
        head = _head_noun(nodes[nid]['label'])
        head_groups[head].append(nid)

    final: Dict[str, List[str]] = {}
    singletons: List[str] = []
    for head, members in head_groups.items():
        if len(members) >= min_size: final[head] = members
        else: singletons.extend(members)

    for nid in singletons:
        label = nodes[nid]['label'].lower()
        best_head = None
        for head in final:
            if head in label: best_head = head; break
        if best_head:
            final[best_head].append(nid)
        elif final:
            largest = max(final, key=lambda h: len(final[h]))
            final[largest].append(nid)
        else:
            final[_head_noun(nodes[nid]['label'])] = [nid]

    while len(final) > max_l2:
        sorted_heads = sorted(final, key=lambda h: len(final[h]))
        small, target = sorted_heads[0], sorted_heads[1]
        final[target].extend(final.pop(small))

    for head in final:
        final[head].sort(key=lambda n: wef_scores.get(n, 0), reverse=True)
    return final

# ─────────────────────────────────────────────────────────────────────────────
# Adjacency list for SPO-driven L3
# ─────────────────────────────────────────────────────────────────────────────

def _build_adjacency(
    edges:    List[Dict],
    nodes:    Dict[str, Dict],
    root_nid: Optional[str],
) -> Dict[str, List[Tuple[str,str,str]]]:
    adj: Dict[str, List[Tuple[str,str,str]]] = defaultdict(list)
    for e in edges:
        s, t, p = e.get('source',''), e.get('target',''), e.get('predicate','related to')
        if not s or not t: continue
        if s != root_nid: adj[s].append((t, p, 'out'))
        if t != root_nid: adj[t].append((s, p, 'in'))
    return adj

def _l3_from_edges(
    l2_nid:  str,
    adj:     Dict[str, List[Tuple[str,str,str]]],
    nodes:   Dict[str, Dict],
    wef:     Dict[str, float],
    used:    Set[str],
    max_l3:  int = 6,
) -> List[Dict[str, Any]]:
    neighbours = adj.get(l2_nid, [])
    neighbours_sorted = sorted(
        neighbours,
        key=lambda x: (0 if x[2]=='out' else 1, -wef.get(x[0], 0))
    )
    seen: Set[str] = set()
    children: List[Dict[str, Any]] = []
    for neighbour_nid, predicate, direction in neighbours_sorted:
        if neighbour_nid in used or neighbour_nid in seen: continue
        if neighbour_nid not in nodes: continue
        seen.add(neighbour_nid)
        children.append({
            'id':        neighbour_nid,
            'label':     _clean(nodes[neighbour_nid]['label']),
            'predicate': predicate,
            'direction': direction,
            'wef_score': round(wef.get(neighbour_nid, 0), 4),
        })
        if len(children) >= max_l3: break
    if not children:
        fallback = [
            nid for nid in sorted(nodes, key=lambda n: wef.get(n,0), reverse=True)
            if nid not in used and nid != l2_nid and nid not in seen
        ]
        for nid in fallback[:max_l3]:
            children.append({
                'id':        nid,
                'label':     _clean(nodes[nid]['label']),
                'predicate': 'related to',
                'direction': 'out',
                'wef_score': round(wef.get(nid, 0), 4),
            })
    return children

# ─────────────────────────────────────────────────────────────────────────────
# Build single hierarchy
# ─────────────────────────────────────────────────────────────────────────────

def _build_one(
    topic_id:       int,
    concept_result: Dict[str, Any],
    lda_result,
    depth:          int,
) -> Dict[str, Any]:
    graph    = concept_result.get('graph', {'nodes':{},'edges':[]})
    nodes    = graph.get('nodes', {})
    edges    = graph.get('edges', [])
    root_nid = graph.get('root_node')

    root_label_str = _root_label(graph, lda_result, topic_id)
    wef_scores     = _wef(graph)
    sorted_ids     = [nid for nid in sorted(wef_scores, key=wef_scores.get, reverse=True)
                      if nid != root_nid]
    root_wef = wef_scores.get(root_nid, 0.0) if root_nid else (
        sum(wef_scores.values()) / len(wef_scores) if wef_scores else 0.0)

    if depth == 1 or not sorted_ids:
        return {'topic_id': topic_id, 'label': root_label_str,
                'wef_score': round(root_wef, 4), 'children': []}

    max_l2   = 5 if depth == 3 else 4
    clusters = _cluster_by_head(sorted_ids, nodes, wef_scores, max_l2=max_l2)
    used_as_l2: Set[str] = set()
    adj = _build_adjacency(edges, nodes, root_nid)

    l2_nodes: List[Dict[str, Any]] = []
    for head, members in sorted(clusters.items(),
                                 key=lambda x: -sum(wef_scores.get(n,0) for n in x[1])):
        best_nid = members[0]
        l2_label = _clean(nodes[best_nid]['label'])
        l2_wef   = sum(wef_scores.get(n, 0) for n in members) / len(members)
        used_as_l2.add(best_nid)

        if depth == 2:
            l2_nodes.append({
                'id': f'l2_{topic_id}_{head}', 'label': l2_label,
                'wef_score': round(l2_wef, 4), 'children': []
            })
        else:
            l3_children = _l3_from_edges(
                l2_nid=best_nid, adj=adj, nodes=nodes,
                wef=wef_scores, used=used_as_l2, max_l3=6,
            )
            # Fill with cluster siblings if SPO edges were sparse
            for sibling_nid in members[1:]:
                if len(l3_children) >= 8: break
                if sibling_nid in used_as_l2: continue
                if any(c['id'] == sibling_nid for c in l3_children): continue
                l3_children.append({
                    'id':        sibling_nid,
                    'label':     _clean(nodes[sibling_nid]['label']),
                    'predicate': 'related to',
                    'direction': 'out',
                    'wef_score': round(wef_scores.get(sibling_nid, 0), 4),
                })
            l2_nodes.append({
                'id': f'l2_{topic_id}_{head}', 'label': l2_label,
                'wef_score': round(l2_wef, 4), 'children': l3_children
            })

    return {'topic_id': topic_id, 'label': root_label_str,
            'wef_score': round(root_wef, 4), 'children': l2_nodes}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    sentence_groups: Dict[int, List[str]],
    concept_results: List[Dict[str, Any]],
    lda_result,
    depth:   int  = 3,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    cr_by_id = {r['topic_id']: r for r in concept_results}
    out = []
    for tid in sorted(sentence_groups):
        cr = cr_by_id.get(tid, {'graph':{'nodes':{},'edges':[]},'sentences':[]})
        h  = _build_one(tid, cr, lda_result, depth)
        if verbose:
            l2 = len(h['children'])
            l3 = sum(len(c['children']) for c in h['children'])
            print(f"  [Phase 4] Topic {tid} root='{h['label']}': "
                  f"L2={l2}  L3={l3}  WEF={h['wef_score']:.3f}")
        out.append(h)
    return out