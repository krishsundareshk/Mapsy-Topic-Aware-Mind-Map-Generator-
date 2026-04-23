"""
pipeline/phase1_preprocessing.py
Phase 1 — Pre-Processing
=========================
Pure-Python implementation mirroring the C pipeline:
  1. Abbreviation expansion
  2. Sentence splitting
  3. Punctuation removal
  4. Basic spelling normalisation (pass-through — C handles true correction)

Input : raw text string
Output: List[str] of clean sentences
"""
from __future__ import annotations
import re
from typing import List

# ─────────────────────────────────────────────────────────────────
# 1. Abbreviation table  (mirrors abbreviation_handler.c)
# ─────────────────────────────────────────────────────────────────
_ABBREV: dict[str, str] = {
    # titles
    "dr.": "Doctor", "mr.": "Mister", "mrs.": "Missus",
    "ms.": "Miss", "prof.": "Professor", "sr.": "Senior", "jr.": "Junior",
    # latin / academic
    "i.e.": "that is", "e.g.": "for example", "etc.": "and so on",
    "vs.": "versus", "et al.": "and others", "viz.": "namely",
    "cf.": "compare", "ibid.": "in the same place",
    # common english
    "approx.": "approximately", "incl.": "including", "excl.": "excluding",
    "max.": "maximum", "min.": "minimum", "avg.": "average",
    "est.": "estimated", "dept.": "department", "govt.": "government",
    "govts.": "governments", "org.": "organisation", "corp.": "corporation",
    "intl.": "international", "natl.": "national", "no.": "number",
    "fig.": "figure", "sec.": "section", "vol.": "volume",
    "yr.": "year", "yrs.": "years", "pct.": "percent", "pct": "percent",
    # domain: case-SENSITIVE uppercase acronyms handled separately
}

_UPPER_ABBREV: dict[str, str] = {
    "CO2":  "carbon dioxide",
    "CH4":  "methane",
    "GHG":  "greenhouse gas",
    "GHGs": "greenhouse gases",
    "IPCC": "Intergovernmental Panel on Climate Change",
    "EV":   "electric vehicle",
    "EVs":  "electric vehicles",
    "ICE":  "internal combustion engine",
    "kWh":  "kilowatt-hours",
    "MW":   "megawatts",
    "GW":   "gigawatts",
    "PV":   "photovoltaic",
}


def _expand_abbreviations(text: str) -> str:
    """Replace known abbreviations with their full forms."""
    tokens = re.split(r'(\s+)', text)
    result = []
    for tok in tokens:
        if tok.strip() == '':
            result.append(tok)
            continue
        # Case-sensitive uppercase first
        if tok in _UPPER_ABBREV:
            result.append(_UPPER_ABBREV[tok])
        else:
            lower = tok.lower()
            if lower in _ABBREV:
                expansion = _ABBREV[lower]
                # Preserve leading capitalisation
                if tok[0].isupper():
                    expansion = expansion[0].upper() + expansion[1:]
                result.append(expansion)
            else:
                result.append(tok)
    return ''.join(result)


# ─────────────────────────────────────────────────────────────────
# 2. Sentence splitting  (mirrors sentence_splitter.c)
# ─────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split text on terminal punctuation (. ! ?) followed by whitespace
    + uppercase, OR on bare newlines.
    """
    sentences: List[str] = []
    # First split on newlines (hard boundaries)
    blocks = [b.strip() for b in text.split('\n') if b.strip()]
    for block in blocks:
        # Within each block, split on '. '  '! '  '? ' before uppercase
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', block)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


# ─────────────────────────────────────────────────────────────────
# 3. Punctuation removal  (mirrors punctuation_removal.c)
# ─────────────────────────────────────────────────────────────────

def _remove_punctuation(sentence: str) -> str:
    """
    Keep: letters, digits, spaces.
    Keep apostrophes between alpha chars (contractions).
    Keep hyphens between word chars (compound words).
    Replace everything else with space, then collapse spaces.
    """
    chars = list(sentence)
    out = []
    n = len(chars)
    for i, c in enumerate(chars):
        prev = chars[i - 1] if i > 0 else ''
        nxt  = chars[i + 1] if i < n - 1 else ''
        if c.isalpha() or c.isdigit():
            out.append(c)
        elif c in (' ', '\t'):
            out.append(' ')
        elif c == "'":
            if prev.isalpha() and nxt.isalpha():
                out.append(c)
            else:
                out.append(' ')
        elif c == '-':
            if (prev.isalpha() or prev.isdigit()) and (nxt.isalpha() or nxt.isdigit()):
                out.append(c)
            else:
                out.append(' ')
        else:
            out.append(' ')
    result = ''.join(out)
    # Collapse spaces
    result = re.sub(r' +', ' ', result).strip()
    return result


# ─────────────────────────────────────────────────────────────────
# 4. Capitalise first letter of each sentence
# ─────────────────────────────────────────────────────────────────

def _capitalise(sentence: str) -> str:
    if not sentence:
        return sentence
    return sentence[0].upper() + sentence[1:]


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def preprocess(raw_text: str, verbose: bool = False) -> List[str]:
    """
    Run all four preprocessing stages on raw_text.

    Parameters
    ----------
    raw_text : str  — raw document text
    verbose  : bool — print progress

    Returns
    -------
    List[str] of clean, one-per-line sentences
    """
    if verbose:
        print(f"[Phase 1] Input: {len(raw_text)} chars")

    # Stage 1 — Abbreviation expansion
    expanded = _expand_abbreviations(raw_text)
    if verbose:
        print(f"[Phase 1] Stage 1 (abbrev expansion) done")

    # Stage 2 — Sentence splitting
    sentences = _split_sentences(expanded)
    if verbose:
        print(f"[Phase 1] Stage 2 (sentence split): {len(sentences)} sentences")

    # Stage 3 — Punctuation removal
    sentences = [_remove_punctuation(s) for s in sentences]
    if verbose:
        print(f"[Phase 1] Stage 3 (punctuation removal) done")

    # Stage 4 — Capitalise + drop empties
    sentences = [_capitalise(s) for s in sentences if s.strip()]
    if verbose:
        print(f"[Phase 1] Stage 4 done. {len(sentences)} clean sentences produced.")

    return sentences


def preprocess_file(filepath: str, verbose: bool = False) -> List[str]:
    """Load a text file and run the full preprocessing pipeline."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    return preprocess(raw, verbose=verbose)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        sents = preprocess_file(path, verbose=True)
    else:
        sample = """The Earth's climate has been changing at an unprecedented rate.
Greenhouse gas emissions, primarily CO2 and CH4, have increased since the industrial revolution.
Renewable energy, e.g. solar and wind, is a critical solution.
EVs offer a transformative solution to reduce carbon emissions approx. 50 pct. by 2040.
Govts. worldwide are offering incentives such as tax credits to accelerate EV adoption."""
        sents = preprocess(sample, verbose=True)
    print("\n=== CLEAN SENTENCES ===")
    for i, s in enumerate(sents):
        print(f"  [{i:02d}] {s}")
