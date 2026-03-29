import json

# ─── Load chunks from JSON ────────────────────────────────────────────────────

def load_chunks(filepath="chunks.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["sentence"], data["chunks"]

# ─── Tree node helpers ────────────────────────────────────────────────────────

def make_node(label, word=None):
    return {
        "label": label,
        "word": word,
        "children": []
    }

def add_child(parent, child):
    parent["children"].append(child)

# ─── Build subtrees ───────────────────────────────────────────────────────────

def build_np(tokens):
    np_node = make_node("NP")
    for token in tokens:
        add_child(np_node, make_node("WORD", token))
    return np_node

def build_pp(tokens):
    pp_node = make_node("PP")
    add_child(pp_node, make_node("WORD", tokens[0]))
    if len(tokens) > 1:
        add_child(pp_node, build_np(tokens[1:]))
    return pp_node

def build_vp(vp_tokens):
    vp_node = make_node("VP")
    for token in vp_tokens:
        add_child(vp_node, make_node("WORD", token))
    return vp_node

# ─── Passive voice detection ──────────────────────────────────────────────────

def is_passive(vp_node):
    words = [c["word"] for c in vp_node["children"] if c["label"] == "WORD"]
    has_aux = any(w.lower() in ["is", "are", "was", "were", "be", "been", "being",
                                 "has", "have", "had"] for w in words)
    has_past_participle = any(w.lower().endswith("ed") for w in words)

    if not (has_aux and has_past_participle):
        return False

    for child in vp_node["children"]:
        if child["label"] == "PP":
            pp_words = [c["word"] for c in child["children"] if c["label"] == "WORD"]
            if pp_words and pp_words[0].lower() == "by":
                return True

    return False

def extract_by_np(vp_node):
    for child in vp_node["children"]:
        if child["label"] == "PP":
            pp_words = [c["word"] for c in child["children"] if c["label"] == "WORD"]
            if pp_words and pp_words[0].lower() == "by":
                for pp_child in child["children"]:
                    if pp_child["label"] == "NP":
                        return pp_child
    return None

def restructure_passive(s_node):
    grammatical_subject = None
    vp_node = None

    for child in s_node["children"]:
        if child["label"] == "NP" and grammatical_subject is None:
            grammatical_subject = child
        elif child["label"] == "VP":
            vp_node = child

    if not vp_node or not is_passive(vp_node):
        return

    semantic_subject = extract_by_np(vp_node)
    if not semantic_subject:
        return

    vp_node["children"] = [
        c for c in vp_node["children"]
        if not (c["label"] == "PP" and
                any(w["word"] and w["word"].lower() == "by"
                    for w in c["children"] if w["label"] == "WORD"))
    ]

    if grammatical_subject:
        add_child(vp_node, grammatical_subject)

    s_node["children"] = [
        semantic_subject if c["label"] == "NP" and c is grammatical_subject
        else c
        for c in s_node["children"]
    ]

# ─── Coordination detection ───────────────────────────────────────────────────

def detect_coordination(chunks):
    # find VP index first
    vp_index = -1
    for i, chunk in enumerate(chunks):
        if chunk["label"] == "VP":
            vp_index = i
            break

    if vp_index == -1:
        return False, []

    # scan pre-VP chunks for NP + OTHER + NP pattern
    co_subjects = []
    i = 0
    while i < vp_index:
        if chunks[i]["label"] == "NP":
            co_subjects.append(chunks[i])
        i += 1

    # coordination only if more than one NP found before VP
    if len(co_subjects) > 1:
        return True, co_subjects

    return False, []

# ─── Build single S tree ──────────────────────────────────────────────────────

def build_single_tree(subject_chunk, chunks, vp_index):
    s_node = make_node("S")

    # build subject
    subject_node = build_np(subject_chunk["tokens"])

    # build VP
    vp_node = build_vp(chunks[vp_index]["tokens"])

    # attach subject and VP to S
    add_child(s_node, subject_node)
    add_child(s_node, vp_node)

    # scan post VP
    last_node = vp_node
    for i in range(vp_index + 1, len(chunks)):
        chunk = chunks[i]
        if chunk["label"] == "NP":
            obj_node = build_np(chunk["tokens"])
            add_child(vp_node, obj_node)
            last_node = obj_node
        elif chunk["label"] == "PP":
            pp_node = build_pp(chunk["tokens"])
            add_child(last_node, pp_node)
            last_node = pp_node
        elif chunk["label"] == "OTHER":
            continue

    restructure_passive(s_node)

    return s_node

# ─── Build full S tree ────────────────────────────────────────────────────────

def build_tree(chunks):
    # check for coordination first
    is_coord, co_subjects = detect_coordination(chunks)

    if is_coord:
        # find VP index
        vp_index = next(i for i, c in enumerate(chunks) if c["label"] == "VP")

        # build one tree per subject
        trees = []
        for subject_chunk in co_subjects:
            tree = build_single_tree(subject_chunk, chunks, vp_index)
            trees.append(tree)
        return trees

    # non-coordinated — build single tree
    s_node = make_node("S")
    subject_node = None
    vp_node = None
    vp_index = -1

    # scan 1 pre-VP
    last_pre_vp_node = None
    for i, chunk in enumerate(chunks):
        if chunk["label"] == "NP" and vp_node is None:
            subject_node = build_np(chunk["tokens"])
            last_pre_vp_node = subject_node
        elif chunk["label"] == "PP" and vp_node is None:
            pp_node = build_pp(chunk["tokens"])
            if last_pre_vp_node:
                add_child(last_pre_vp_node, pp_node)
            last_pre_vp_node = pp_node
        elif chunk["label"] == "VP" and vp_node is None:
            vp_node = build_vp(chunk["tokens"])
            vp_index = i
            break

    if subject_node:
        add_child(s_node, subject_node)
    if vp_node:
        add_child(s_node, vp_node)

    # scan 2 post-VP
    if vp_index != -1:
        last_node = vp_node
        for i in range(vp_index + 1, len(chunks)):
            chunk = chunks[i]
            if chunk["label"] == "NP":
                obj_node = build_np(chunk["tokens"])
                add_child(vp_node, obj_node)
                last_node = obj_node
            elif chunk["label"] == "PP":
                pp_node = build_pp(chunk["tokens"])
                add_child(last_node, pp_node)
                last_node = pp_node
            elif chunk["label"] == "OTHER":
                continue

    restructure_passive(s_node)

    return [s_node]  # always return a list for consistency

# ─── Extract SPO from single tree ────────────────────────────────────────────

def extract_spo(tree):
    subject = None
    predicate = None
    obj = None

    for child in tree["children"]:
        if child["label"] == "NP" and subject is None:
            subject = " ".join(
                c["word"] for c in child["children"] if c["word"]
            )
        elif child["label"] == "VP":
            vp_words = []
            for vp_child in child["children"]:
                if vp_child["label"] == "WORD":
                    vp_words.append(vp_child["word"])
                elif vp_child["label"] == "NP":
                    obj = " ".join(
                        c["word"] for c in vp_child["children"] if c["word"]
                    )
            predicate = " ".join(vp_words) if vp_words else None

    return subject, predicate, obj

# ─── Print tree ───────────────────────────────────────────────────────────────

def print_tree(node, indent=0):
    prefix = "  " * indent
    if node["word"]:
        print(f"{prefix}[{node['label']}] '{node['word']}'")
    else:
        print(f"{prefix}[{node['label']}]")
    for child in node["children"]:
        print_tree(child, indent + 1)

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("chunks.json", "r") as f:
        data = json.load(f)

    for topic in data["topics"]:
        print(f"\n── Topic {topic['topic_id']} ──")

        for sentence in topic["sentences"]:
            print(f"\n  Sentence [{sentence['id']}]: {sentence['text'][:60]}...")

            trees = build_tree(sentence["chunks"])

            for i, tree in enumerate(trees):
                subject, predicate, obj = extract_spo(tree)
                if subject and predicate:
                    print(f"    SPO: ({subject}, {predicate}, {obj if obj else 'None'})")
