"""
pipeline/phase3_concept_extraction.py — v3
============================================
Changes from v2
---------------
1. GENERIC_PREDICATES blocklist — verbs like "include", "contain",
   "involve", "comprise" are now skipped in _pick_verb_in_window so they
   never become predicate labels.  If no real content verb is found the
   edge falls back to "related to" (dim rendering in Phase 5) rather than
   flooding every arrow with "include".

2. Everything else (SPO extraction, TF-IDF scoring, junk filtering,
   degree-centrality root selection) is identical to v2.
"""
from __future__ import annotations
import re, math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Set, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Word-category sets
# ─────────────────────────────────────────────────────────────────────────────

PRONOUNS = {
    'i','me','my','mine','myself','we','us','our','ours','ourselves',
    'you','your','yours','yourself','he','him','his','himself',
    'she','her','hers','herself','it','its','itself',
    'they','them','their','theirs','themselves','one','ones',
}
DETERMINERS = {
    'a','an','the','this','that','these','those','each','every',
    'any','all','both','no','some','such','my','your','its','our',
    'another','other','others','same','different',
}
PREPOSITIONS = {
    'in','on','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below',
    'from','up','down','of','off','over','under','toward','upon',
    'to','than','across','within','throughout','around','along',
    'among','per','via','without','except','plus','like','unlike',
    'despite','regarding','concerning','considering','excluding',
    'beyond','beside','besides','behind','ahead','inside','outside',
}
CONJUNCTIONS = {
    'and','but','or','nor','yet','so','because','although','while',
    'where','when','if','as','that','which','who','whom','whose',
    'whether','since','until','unless','after','before','though',
    'even','neither','either','however','therefore','thus','hence',
    'moreover','furthermore','meanwhile','otherwise','consequently',
}
MODALS      = {'can','could','will','would','shall','should','may','might','must','need','ought'}
AUXILIARIES = {'is','are','was','were','be','been','am','have','has','had','do','does','did','get','got','being'}

ADVERBS = {
    'very','quite','rather','too','so','just','already','still','also',
    'even','only','never','ever','often','always','usually','rapidly',
    'significantly','dramatically','substantially','largely','primarily',
    'mainly','generally','particularly','especially','widely','globally',
    'severely','urgently','critically','currently','increasingly',
    'continually','approximately','efficiently','effectively','consistently',
    'continuously','nearly','almost','entirely','mostly','highly','deeply',
    'greatly','strongly','clearly','simply','easily','quickly','slowly',
    'fully','heavily','previously','recently','specifically','directly',
    'closely','roughly','alone','together','apart','ahead','behind',
    'instead','otherwise','meanwhile','overall','moreover','furthermore',
    'nevertheless','nonetheless','accordingly','consequently','therefore',
    'thus','hence','thereby','whereby','wherein','therein','herein',
}
ADJECTIVES = {
    'new','old','good','bad','big','small','large','high','low','great',
    'major','global','local','natural','social','economic','clean','clear',
    'direct','fast','slow','strong','weak','safe','stable','consistent',
    'efficient','effective','sustainable','renewable','reliable','resilient',
    'diverse','complex','simple','advanced','modern','innovative',
    'industrial','commercial','coastal','unprecedented','transformative',
    'vulnerable','severe','extreme','alarming','widespread','primary',
    'secondary','initial','final','total','entire','common','public',
    'private','key','electric','solar','wind','fossil','thermal','kinetic',
    'potential','chemical','nuclear','polar','geothermal','photovoltaic',
    'smart','rechargeable','solid','lithium','carbon','greenhouse','ancient',
    'powerful','political','military','cultural','religious','historical',
    'traditional','democratic','scientific','biological','physical',
    'molecular','cellular','genetic','ecological','digital','artificial',
    'autonomous','connected','shared','rapid','critical','essential',
    'necessary','possible','available','current','recent','future','past',
    'present','next','last','early','late','first','second','third','main',
    'central','western','eastern','northern','southern','upper','lower',
    'inner','outer','deep','broad','narrow','long','short','heavy','light',
    'hot','cold','warm','cool','wet','dry','hard','soft','open','closed',
    'full','empty','rich','poor','free','fresh','raw','urban','rural',
    'internal','external','international','national','regional','official',
    'formal','legal','basic','fundamental','comprehensive','specific',
    'general','unique','special','standard','normal','regular','typical',
    'significant','important','vital','leading','growing','rising','falling',
    'increasing','decreasing','expanding','contracting','emerging','existing',
    'multiple','various','several','certain','particular','respective',
    'overall','substantial','considerable','notable','remarkable',
}
KNOWN_NOUNS = {
    'climate','warming','temperature','emission','emissions','carbon',
    'dioxide','methane','fossil','fuel','fuels','deforestation','glacier',
    'glaciers','sea','level','levels','weather','hurricane','drought',
    'wildfire','food','biodiversity','habitat','species','energy','power',
    'electricity','grid','battery','batteries','storage','turbine',
    'turbines','panel','panels','vehicle','vehicles','motor','motors',
    'engine','engines','transportation','infrastructure','technology',
    'technologies','automaker','automakers','incentive','incentives',
    'credit','credits','adoption','mobility','rate','cost','costs',
    'percent','decade','century','world','community','communities',
    'government','governments','country','countries','region','regions',
    'industry','market','economy','data','research','source','sources',
    'solution','system','systems','network','station','stations','range',
    'earth','atmosphere','ecosystem','planet','sun','wind','water','heat',
    'light','process','reaction','molecule','cell','cells','plant',
    'plants','oxygen','photosynthesis','chlorophyll','chloroplast',
    'thylakoid','stroma','glucose','sugar','protein','enzyme','pigment',
    'leaf','leaves','root','roots','stem','flower','crop','crops',
    'empire','republic','senate','army','military','territory','province',
    'civilization','culture','religion','law','trade','commerce','road',
    'algorithm','model','computation','intelligence','machine','robot',
    'software','hardware','memory','processor','nation','state','city',
    'population','growth','development','production','consumption',
    'pollution','pressure','concentration','distribution',
    'impact','effect','change','increase','decrease','reduction',
    'transition','transformation','revolution','evolution',
    'deployment','integration','implementation','application','operation',
}
KNOWN_VERBS = {
    'change','changes','changed','cause','causes','caused',
    'increase','increases','increased','decrease','decreases','decreased',
    'reduce','reduces','reduced','generate','generates','generated',
    'provide','provides','provided','use','uses','used',
    'convert','converts','converted','harness','harnesses','harnessed',
    'affect','affects','affected','offer','offers','offered',
    'enable','enables','enabled','create','creates','created',
    'accelerate','expand','expands','expanded','install','installs','installed',
    'announce','announces','announced','shift','shifts','shifted',
    'emit','emits','emitted','produce','produces','produced',
    'supply','supplies','supplied','support','supports','supported',
    'protect','protects','protected','prevent','prevents','prevented',
    'displace','displaces','displaced','disrupt','disrupts','disrupted',
    'adopt','adopts','adopted','tap','taps','tapped','power','powers','powered',
    'drive','drives','driven','rise','rises','risen',
    'grow','grows','grown','burn','burns','burned','melt','melts','melted',
    'warm','warms','warmed','store','stores','stored',
    'charge','charges','charged','travel','travels','traveled',
    'threaten','threatens','threatened','warn','warns','warned',
    'drop','drops','dropped','invest','invests','invested',
    'develop','develops','developed','advance','advances','advanced',
    'improve','improves','improved','manage','manages','managed',
    'replace','replaces','replaced','begin','begins','began',
    'continue','continues','continued','transition','transitions','transitioned',
    'contribute','contributes','contributed',
    'control','controls','controlled','allow','allows','allowed',
    'require','requires','required','include','includes','included',
    'contain','contains','contained','form','forms','formed',
    'build','builds','built','make','makes','made','take','takes','taken',
    'give','gives','given','come','comes','came','go','goes','went',
    'see','sees','saw','think','thinks','thought',
    'become','becomes','became','show','shows','showed',
    'keep','keeps','kept','hold','holds','held','turn','turns','turned',
    'run','runs','ran','help','helps','helped',
    'work','works','worked','move','moves','moved','lead','leads','led',
    'play','plays','played','reach','reaches','reached',
    'span','spans','spanned','stand','stands','stood',
    'remain','remains','remained','fall','falls','fell',
    'spread','spreads','serve','serves','served','rule','rules','ruled',
    'govern','governs','governed','fight','fights','fought',
    'conquer','conquers','conquered','defeat','defeats','defeated',
    'establish','establishes','established','found','founds','founded',
    'occur','occurs','occurred','perform','performs','performed',
    'absorb','absorbs','absorbed','fix','fixes','fixed',
    'release','releases','released','capture','captures','captured',
    'reflect','reflects','reflected','conduct','conducts','conducted',
    'react','reacts','reacted','combine','combines','combined',
    'separate','separates','separated','divide','divides','divided',
    'evolve','evolves','evolved','adapt','adapts','adapted',
    'survive','survives','survived','deploy','deploys','deployed',
    'observe','observes','observed','discover','discovers','discovered',
    'apply','applies','applied','test','tests','tested',
    'refer','refers','referred','represents','integrate','integrates','integrated',
    'spin','spins','spun','follow','follows','followed',
    'limit','limits','limited','link','links','linked',
    'connect','connects','connected','relate','relates','related',
    'depend','depends','depended','involve','involves','involved',
    'determine','determines','determined','define','defines','defined',
    'measure','measures','measured','calculate','calculates','calculated',
    'estimate','estimates','estimated','predict','predicts','predicted',
    'explain','explains','explained','describe','describes','described',
    'indicate','indicates','indicated','suggest','suggests','suggested',
    'demonstrate','demonstrates','demonstrated','confirm','confirms','confirmed',
    'reveal','reveals','revealed',
    'highlight','highlights','highlighted','emphasize','emphasizes',
    'promote','promotes','promoted','facilitate','facilitates','facilitated',
    'inhibit','inhibits','inhibited',
    'block','blocks','blocked',
    'trigger','triggers','triggered','initiate','initiates','initiated',
    'terminate','terminates','terminated','transform','transforms','transformed',
    'regulate','regulates','regulated','monitor','monitors','monitored',
    'track','tracks','tracked',
}

# ── NEW v3: generic composition verbs — never used as predicate labels ───────
_GENERIC_PREDICATES = {
    'include','includes','included','including',
    'contain','contains','contained','containing',
    'involve','involves','involved','involving',
    'comprise','comprises','comprised','comprising',
    'consist','consists','consisted','consisting',
    'represent','represents','represented','representing',
    'form','forms','formed','forming',
    'make','makes','made','making',
    'take','takes','took','taking',
    'give','gives','gave','giving',
    'refer','refers','referred','referring',
    'come','comes','came','coming',
    'go','goes','went','going',
    'get','gets','got','getting',
    'have','has','had','having',
    'keep','keeps','kept','keeping',
    'remain','remains','remained','remaining',
    'become','becomes','became','becoming',
}

_JUNK_NODES = {
    'many','much','more','most','less','least','few','several','various',
    'numerous','multiple','some','any','all','both','each','every',
    'enough','plenty','lot','lots','bit','little','quite','rather',
    'very','too','so','just','only','even','still','already','yet',
    'once','twice','often','always','never','sometimes','usually',
    'generally','typically','approximately','about','around','nearly',
    'almost','exactly','precisely','roughly','largely','mainly',
    'mostly','primarily','particularly','especially','specifically',
    'broadly','widely','deeply','greatly','highly',
    'this','that','these','those','such','same','different','other',
    'another','others','both','either','neither','certain','given',
    'particular','respective','overall','general','specific','various',
    'way','thing','things','part','parts','type','types','kind','kinds',
    'form','forms','case','cases','use','uses','work','works',
    'life','day','days','time','times','place','places','point','points',
    'fact','facts','rate','rates','level','levels',
    'amount','amounts','number','numbers','result','results',
    'area','areas','side','sides','end','ends','set','sets',
    'group','groups','list','lists','item','items','step','steps',
    'line','lines','term','terms','word','words','name','names',
    'idea','ideas','problem','problems','question','questions',
    'answer','answers','reason','reasons','cause','causes',
    'effect','effects','impact','impacts','period','periods',
    'moment','moments','sense','senses','view','views','role','roles',
    'need','needs','goal','goals','aim','aims','plan','plans',
    'action','actions','effort','efforts','piece','pieces',
    'section','sections','version','versions','base','bases',
    'world','earth','planet',
    'example','examples','instance','instances','sample','samples',
    'scenario','scenarios','situation','situations',
    'percent','percentage','ratio','proportion','fraction',
    'degree','degrees','scale','scales','range','ranges',
    'limit','limits','threshold','thresholds',
    'single','one','two','three','four','five','six','seven',
    'eight','nine','ten','hundred','thousand','million','billion',
    'sunlight','sunlights','frequent','frequently','unprecedented',
    'unprecedented rate','past century','modern electricity',
}

# ─────────────────────────────────────────────────────────────────────────────
# POS tagger
# ─────────────────────────────────────────────────────────────────────────────

def _tag(word: str) -> str:
    w = word.lower()
    if re.fullmatch(r'[0-9]+(\.[0-9]+)?', w): return 'CD'
    if w in PRONOUNS:    return 'PR'
    if w in DETERMINERS: return 'DT'
    if w in MODALS:      return 'MD'
    if w in AUXILIARIES: return 'VB'
    if w in PREPOSITIONS:return 'IN'
    if w in CONJUNCTIONS:return 'CC'
    if w in ADVERBS:     return 'RB'
    if w in ADJECTIVES:  return 'JJ'
    if w in KNOWN_VERBS: return 'VB'
    if w in KNOWN_NOUNS: return 'NN'
    if w.endswith('ly'): return 'RB'
    if w.endswith(('tion','tions','ment','ments','ness','ity','ism',
                   'ist','age','ance','ence','ship','hood','dom',
                   'ics','ogy','ery','ary','ory')): return 'NN'
    if w.endswith(('ed','ing')): return 'VB'
    if w.endswith(('al','ful','ous','ive','ble','ic','ish')): return 'JJ'
    if len(w) > 2 and word[0].isupper() and not word.isupper(): return 'NN'
    if w.endswith('s') and len(w) > 4: return 'NN'
    return 'NN'

def _pos_tag(sentence: str) -> List[Tuple[str, str]]:
    tokens = re.findall(r"[a-zA-Z][\w'\-]*|[0-9]+(?:\.[0-9]+)?", sentence)
    return [(t, _tag(t)) for t in tokens]

# ─────────────────────────────────────────────────────────────────────────────
# Noun chunk extraction
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_START = PRONOUNS | DETERMINERS | CONJUNCTIONS | PREPOSITIONS | ADVERBS | AUXILIARIES | MODALS | KNOWN_VERBS

def _strip_trailing(words: List[str]) -> List[str]:
    _trail = ADVERBS | PREPOSITIONS | CONJUNCTIONS | {
        'well','like','alone','together','only','just','even','also','very','quite',
    }
    while words and words[-1].lower() in _trail: words = words[:-1]
    while words and words[0].lower()  in _trail: words = words[1:]
    return words

def _extract_chunks(sentence: str) -> List[str]:
    tagged = _pos_tag(sentence)
    chunks: List[str] = []
    i, n = 0, len(tagged)
    while i < n:
        word, tag = tagged[i]
        if tag in ('PR','VB','IN','CC','RB','MD','CD'): i += 1; continue
        start = i
        if tag == 'DT':
            i += 1
            if i >= n: break
            word, tag = tagged[i]
            if tag in ('PR','VB','IN','CC','RB','MD'): continue
        while i < n and tagged[i][1] == 'JJ': i += 1
        if i < n and tagged[i][1] == 'NN':
            while i < n and tagged[i][1] == 'NN': i += 1
            tokens = [w for w, t in tagged[start:i]
                      if t != 'DT' and w.lower() not in _SKIP_START]
            if tokens:
                words = _strip_trailing(tokens)
                if not words: continue
                if words[0].lower() in KNOWN_VERBS: continue
                if words[0].lower() in PRONOUNS: continue
                chunk = ' '.join(words).lower()
                chunks.append(chunk)
        else:
            i = max(i, start + 1)
    for k in range(len(tagged) - 1):
        w1, t1 = tagged[k]; w2, t2 = tagged[k+1]
        if (t1 == 'NN' and t2 == 'NN'
                and w1.lower() not in _SKIP_START
                and w2.lower() not in _SKIP_START):
            phrase = f'{w1} {w2}'.lower()
            if phrase not in chunks: chunks.append(phrase)
    return list(dict.fromkeys(chunks))

def _is_good(chunk: str) -> bool:
    words = chunk.split()
    if not words or len(chunk) < 4: return False
    if chunk in _JUNK_NODES: return False
    if all(w in _JUNK_NODES for w in words): return False
    if all(w in _SKIP_START for w in words): return False
    if all(w.isdigit() for w in words): return False
    if words[0] in KNOWN_VERBS or words[0] in PRONOUNS: return False
    if len(words) == 1 and words[0] in ADJECTIVES and words[0] not in KNOWN_NOUNS: return False
    if len(words) == 1 and words[0] in _JUNK_NODES: return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF scorer
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf(chunks_per_sent: List[List[str]]) -> Dict[str, float]:
    n = len(chunks_per_sent) or 1
    df: Counter = Counter()
    for c in chunks_per_sent:
        for w in set(c): df[w] += 1
    scores: Dict[str, float] = {}
    for doc in chunks_per_sent:
        total = len(doc) or 1
        for w, freq in Counter(doc).items():
            scores[w] = scores.get(w, 0.0) + (freq / total) * (math.log((1 + n) / (1 + df[w])) + 1)
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# SPO triple extractor
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_verb(verb: str) -> str:
    v = verb.lower().rstrip('.')
    for suffix, replace in [
        ('ies','y'),('ied','y'),('ves','ve'),('ses','se'),
        ('zes','ze'),('xes','x'),('ches','ch'),('shes','sh'),
        ('ing',''),('ed',''),('es',''),('s',''),
    ]:
        if v.endswith(suffix) and len(v) - len(suffix) >= 3:
            return v[:-len(suffix)] + replace
    return v

def _find_span(words: List[str], chunk: str) -> Tuple[int, int]:
    cw = chunk.lower().split()
    n  = len(words)
    for start in range(n - len(cw) + 1):
        if words[start:start + len(cw)] == cw:
            return start, start + len(cw)
    head = cw[-1]
    for start in range(n):
        if words[start] == head: return start, start + 1
    return -1, -1

_PURE_AUX = {
    'is','are','was','were','be','been','am','being',
    'have','has','had','do','does','did','get','got',
}

def _pick_verb_in_window(tagged: List[Tuple[str,str]], start: int, end: int) -> str:
    """Return first content verb in tagged[start:end], skipping pure auxiliaries
    AND generic composition verbs (v3 addition)."""
    for idx in range(max(0, start), min(len(tagged), end)):
        w, t = tagged[idx]
        wl = w.lower()
        if t in ('VB','MD') and wl not in _PURE_AUX and wl not in _GENERIC_PREDICATES:
            return _normalise_verb(w)
    return ''

def _extract_spo_triples(sentence: str, concept_set: Set[str]) -> List[Tuple[str,str,str]]:
    tagged = _pos_tag(sentence)
    words  = [w.lower() for w, _ in tagged]
    present: List[Tuple[str,int,int]] = []
    for concept in concept_set:
        s, e = _find_span(words, concept)
        if s != -1: present.append((concept, s, e))
    present.sort(key=lambda x: x[1])
    triples: List[Tuple[str,str,str]] = []
    seen_pairs: Set[Tuple[str,str]] = set()
    for i in range(len(present)):
        for j in range(i+1, min(i+4, len(present))):
            a_concept, a_start, a_end = present[i]
            b_concept, b_start, b_end = present[j]
            if a_concept == b_concept: continue
            pair = (a_concept, b_concept)
            if pair in seen_pairs: continue
            seen_pairs.add(pair)
            verb = _pick_verb_in_window(tagged, a_end, b_start)
            if not verb:
                verb = _pick_verb_in_window(tagged, max(a_end, b_start-5), b_start)
            if not verb:
                verb = _pick_verb_in_window(tagged, a_end, min(a_end+6, b_start+1))
            if verb:
                triples.append((a_concept, verb, b_concept))
    return triples

# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph(sentences: List[str], top_n: int = 20) -> Dict[str, Any]:
    cps  = [_extract_chunks(s) for s in sentences]
    raw  = _tfidf(cps)
    good = {k: v for k, v in raw.items() if _is_good(k)}
    top  = sorted(good, key=good.get, reverse=True)[:top_n]
    cset = set(top)

    nodes: Dict[str,Dict] = {}
    c2id:  Dict[str,str]  = {}
    for i, c in enumerate(top):
        nid = f'n{i}'
        nodes[nid] = {'label': c, 'freq': round(good[c], 4)}
        c2id[c] = nid

    triple_counts: Dict[Tuple[str,str,str],int] = Counter()
    for sent in sentences:
        for triple in _extract_spo_triples(sent, cset):
            triple_counts[triple] += 1

    edges: List[Dict]               = []
    seen_pairs: Set[Tuple[str,str]] = set()

    for (subj, verb, obj), count in sorted(triple_counts.items(), key=lambda x: -x[1]):
        sid = c2id.get(subj); oid = c2id.get(obj)
        if sid is None or oid is None: continue
        pair = (sid, oid)
        if pair in seen_pairs: continue
        seen_pairs.add(pair)
        edges.append({'source': sid, 'target': oid, 'predicate': verb, 'weight': count})

    pair_sentences: Dict[Tuple[str,str],List[str]] = defaultdict(list)
    for sent, chunks in zip(sentences, cps):
        present = [c for c in chunks if c in cset]
        for ai in range(len(present)):
            for bi in range(ai+1, min(ai+3, len(present))):
                c1, c2 = present[ai], present[bi]
                if c1 != c2: pair_sentences[(c1, c2)].append(sent)

    for (c1, c2), sents in pair_sentences.items():
        sid = c2id.get(c1); oid = c2id.get(c2)
        if sid is None or oid is None: continue
        pair = (sid, oid); rev = (oid, sid)
        if pair in seen_pairs or rev in seen_pairs: continue
        seen_pairs.add(pair)
        edges.append({'source': sid, 'target': oid, 'predicate': 'related to', 'weight': len(sents)})

    return {'nodes': nodes, 'edges': edges}

# ─────────────────────────────────────────────────────────────────────────────
# Root-node selection
# ─────────────────────────────────────────────────────────────────────────────

def _pick_root_node(graph: Dict[str, Any]) -> Optional[str]:
    if not graph['edges']: return 'n0' if graph['nodes'] else None
    degree: Dict[str,float] = defaultdict(float)
    for edge in graph['edges']:
        w = edge.get('weight', 1)
        degree[edge['source']] += w
        degree[edge['target']] += w
    for nid, info in graph['nodes'].items():
        degree[nid] = degree.get(nid, 0.0) + info['freq'] * 0.5
    return max(degree, key=degree.__getitem__)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_theme(topic_id: int, sentences: List[str], verbose: bool = False) -> Dict[str, Any]:
    graph = _build_graph(sentences, top_n=20)
    graph['root_node'] = _pick_root_node(graph)
    if verbose:
        spo_count  = sum(1 for e in graph['edges'] if e['predicate'] != 'related to')
        fall_count = len(graph['edges']) - spo_count
        root_label = (graph['nodes'].get(graph['root_node'], {}).get('label', '?')
                      if graph['root_node'] else '?')
        print(f"  [Phase 3] Topic {topic_id}: {len(sentences)} sents "
              f"→ {len(graph['nodes'])} concepts, {len(graph['edges'])} edges "
              f"({spo_count} SPO-derived, {fall_count} co-occurrence fallback) "
              f"| root → '{root_label}'")
    return {'topic_id': topic_id, 'graph': graph, 'sentences': sentences}

def run(sentence_groups: Dict[int, List[str]], verbose: bool = False) -> List[Dict[str, Any]]:
    return [process_theme(tid, sentence_groups[tid], verbose) for tid in sorted(sentence_groups)]