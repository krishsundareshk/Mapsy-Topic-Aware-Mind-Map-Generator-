/* =============================================================
 * punctuation_removal.h
 * Removes punctuation characters from each sentence in a
 * SentenceList, operating in-place on the list.
 * ============================================================= */

#ifndef PUNCTUATION_REMOVAL_H
#define PUNCTUATION_REMOVAL_H

#include "preprocessing_types.h"

/* ---------------------------------------------------------------
 * remove_punctuation_from_list()
 *
 * Iterates over every sentence in 'sl' and strips punctuation.
 * Operates in-place: the same SentenceList is modified.
 *
 * Rules applied
 *   - Standard punctuation ( . , ! ? ; : " ' ( ) [ ] { } /
 *     \ @ # $ % ^ & * - _ + = < > ~ ` | ) is removed.
 *   - An apostrophe INSIDE a word (contractions / possessives)
 *     is preserved  e.g.  "it's" → "it's"  but  '"hello"' → 'hello'.
 *   - Multiple consecutive spaces that arise after removal are
 *     collapsed into a single space.
 *   - Leading and trailing spaces are stripped.
 *
 * Returns PP_OK on success, PP_ERR_NULL if sl is NULL.
 * --------------------------------------------------------------- */
int remove_punctuation_from_list(SentenceList *sl);

/* ---------------------------------------------------------------
 * remove_punctuation_from_sentence()
 *
 * Applies the same rules to a single null-terminated string,
 * writing the result into 'output' (caller-allocated,
 * size MAX_SENTENCE_LEN).
 *
 * Exposed so it can be unit-tested independently.
 * --------------------------------------------------------------- */
int remove_punctuation_from_sentence(const char *input, char *output);

#endif /* PUNCTUATION_REMOVAL_H */
