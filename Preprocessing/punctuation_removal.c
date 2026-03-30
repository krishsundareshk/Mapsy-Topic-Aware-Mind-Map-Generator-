/* =============================================================
 * punctuation_removal.c
 *
 * Algorithm for a single sentence
 * --------------------------------
 * Walk character by character:
 *   - Letters and digits   → always copy.
 *   - Space                → copy (will be de-duplicated later).
 *   - Apostrophe           → copy only if sandwiched between
 *                            two alphabetic characters (contraction
 *                            or possessive); otherwise discard.
 *   - Hyphen '-'           → copy only if sandwiched between
 *                            two word characters (compound words);
 *                            otherwise discard (e.g. bullet dashes).
 *   - All other punctuation → replace with a space so that
 *                            adjacent words do not merge.
 *
 * After the pass, collapse runs of spaces into a single space
 * and strip leading/trailing whitespace.
 * ============================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "punctuation_removal.h"
#include "preprocessing_types.h"

/* ---- helper: collapse multiple spaces in-place ---- */
static void collapse_spaces(char *s) {
    char *r = s, *w = s;
    int   in_space = 0;

    while (*r) {
        if (isspace((unsigned char)*r)) {
            if (!in_space) { *w++ = ' '; in_space = 1; }
        } else {
            *w++ = *r;
            in_space = 0;
        }
        r++;
    }
    *w = '\0';

    /* strip leading/trailing */
    int len = (int)strlen(s);
    if (len > 0 && s[0] == ' ') memmove(s, s + 1, len--);
    if (len > 0 && s[len - 1] == ' ') s[--len] = '\0';
}

/* ================================================================
 * remove_punctuation_from_sentence()
 * ================================================================ */
int remove_punctuation_from_sentence(const char *input, char *output) {
    if (!input || !output) return PP_ERR_NULL;

    const char *src = input;
    char       *dst = output;
    int         ilen = (int)strlen(input);

    for (int i = 0; i < ilen; i++) {
        unsigned char c    = (unsigned char)src[i];
        unsigned char prev = (i > 0)       ? (unsigned char)src[i - 1] : '\0';
        unsigned char next = (i < ilen - 1) ? (unsigned char)src[i + 1] : '\0';

        if (isalpha(c) || isdigit(c)) {
            /* alphanumeric: always keep */
            *dst++ = (char)c;

        } else if (c == ' ' || c == '\t') {
            /* whitespace: keep one space */
            *dst++ = ' ';

        } else if (c == '\'') {
            /* apostrophe: keep only inside alpha-alpha (contractions) */
            if (isalpha(prev) && isalpha(next))
                *dst++ = '\'';
            else
                *dst++ = ' ';

        } else if (c == '-') {
            /* hyphen: keep inside word-word (compound adjectives etc.) */
            if ((isalpha(prev) || isdigit(prev)) &&
                (isalpha(next) || isdigit(next)))
                *dst++ = '-';
            else
                *dst++ = ' ';

        } else {
            /* every other punctuation char → space to separate words */
            *dst++ = ' ';
        }
    }
    *dst = '\0';

    collapse_spaces(output);
    return PP_OK;
}

/* ================================================================
 * remove_punctuation_from_list()
 * ================================================================ */
int remove_punctuation_from_list(SentenceList *sl) {
    if (!sl) return PP_ERR_NULL;

    char buffer[MAX_SENTENCE_LEN];

    for (int i = 0; i < sl->count; i++) {
        int ret = remove_punctuation_from_sentence(sl->data[i], buffer);
        if (ret != PP_OK) continue;           /* skip on error */
        strncpy(sl->data[i], buffer, MAX_SENTENCE_LEN - 1);
        sl->data[i][MAX_SENTENCE_LEN - 1] = '\0';
    }

    return PP_OK;
}
