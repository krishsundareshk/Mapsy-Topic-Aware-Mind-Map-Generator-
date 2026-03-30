/* =============================================================
 * spelling_correction.h
 * Lightweight spelling correction using Levenshtein distance
 * against a built-in and/or file-loaded word dictionary.
 * ============================================================= */

#ifndef SPELLING_CORRECTION_H
#define SPELLING_CORRECTION_H

#include "preprocessing_types.h"

#define DICT_MAX_WORDS   8192
#define DICT_WORD_LEN    64

/* Only correct if edit distance <= this value.
 * 1 = conservative (catches single-char typos only, avoids false positives) */
#define MAX_EDIT_DISTANCE 1

int         sc_load_dictionary(const char *dict_filepath);
const char *sc_correct_word(const char *word);
int         correct_spelling_in_list(SentenceList *sl);
void        sc_print_dictionary(void);

#endif /* SPELLING_CORRECTION_H */
