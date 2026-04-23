"""
pipeline/node_lemmatizer.py  (patched — no NLTK data required)
===============================================================
Drop-in replacement for the original node_lemmatizer.py.
Uses suffix-stripping rules instead of WordNet so it works
without any downloaded NLTK corpora.
"""
from __future__ import annotations
import re
from typing import List, Tuple

SPOTriplet      = Tuple[str, str, str]
LemmatisedTuple = Tuple[str, str, str]

_NOUN_RULES = [
    ('criteria','criterion'),('phenomena','phenomenon'),
    ('data','datum'),('media','medium'),
    ('indices','index'),('vertices','vertex'),
    ('matrices','matrix'),('analyses','analysis'),
    ('bases','basis'),('crises','crisis'),
    ('theses','thesis'),('axes','axis'),
    ('gases','gas'),('buses','bus'),
]
_NOUN_SUFFIXES = [
    ('ies','y'),('ves','f'),('xes','x'),('ses','s'),
    ('ches','ch'),('shes','sh'),('zzes','z'),('sses','ss'),
    ('s',''),
]
_MIN_STEM = 3

def lemmatise_word(word: str) -> str:
    w = word.lower().strip()
    if len(w) <= 3:
        return w
    for plural, singular in _NOUN_RULES:
        if w == plural:
            return singular
    for suffix, replacement in _NOUN_SUFFIXES:
        if w.endswith(suffix):
            stem = w[:len(w)-len(suffix)] + replacement
            if len(stem) >= _MIN_STEM:
                return stem
    return w

def lemmatise_phrase(phrase: str) -> str:
    phrase  = re.sub(r"[^\w\s-]", "", phrase)
    tokens  = phrase.lower().split()
    stopdet = {'the','a','an','this','that','these','those'}
    tokens  = [t for t in tokens if t not in stopdet and not t.isdigit()]
    if not tokens:
        return phrase.lower()
    if len(tokens) == 1:
        return lemmatise_word(tokens[0])
    *modifiers, head = tokens
    return ' '.join(modifiers + [lemmatise_word(head)])

def lemmatise_triplet(triplet):
    subject, predicate, obj = triplet
    return (
        lemmatise_phrase(subject),
        predicate.lower().strip(),
        lemmatise_phrase(obj) if obj else '',
    )

def lemmatise_all(triplets):
    return [lemmatise_triplet(t) for t in triplets]
