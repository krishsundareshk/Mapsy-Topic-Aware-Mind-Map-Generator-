/* =============================================================
 * sentence_splitter.h
 * Splits a preprocessed (abbreviation-expanded) text block
 * into individual sentences and stores them in a SentenceList.
 * ============================================================= */

#ifndef SENTENCE_SPLITTER_H
#define SENTENCE_SPLITTER_H

#include "preprocessing_types.h"

/* ---------------------------------------------------------------
 * split_sentences()
 *
 * Tokenises 'text' into sentences using the heuristic:
 *   A sentence boundary occurs at  '.', '!', or '?'  when it is
 *   followed by whitespace and an uppercase letter (or end of
 *   string).  Consecutive whitespace-only "sentences" are
 *   discarded.
 *
 * Parameters
 *   text   : null-terminated input string (abbreviation-expanded)
 *   out    : caller-allocated SentenceList; out->count is set.
 *
 * Returns  PP_OK on success, error code otherwise.
 * --------------------------------------------------------------- */
int split_sentences(const char *text, SentenceList *out);

/* ---------------------------------------------------------------
 * print_sentence_list()
 * Debug helper — prints all sentences with their index.
 * --------------------------------------------------------------- */
void print_sentence_list(const SentenceList *sl);

#endif /* SENTENCE_SPLITTER_H */
