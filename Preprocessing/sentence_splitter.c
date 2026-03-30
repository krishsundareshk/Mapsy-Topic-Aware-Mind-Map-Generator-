/* =============================================================
 * sentence_splitter.c
 *
 * Heuristic sentence boundary detection.
 *
 * Design
 * ------
 * After abbreviation expansion (which removes all dots from
 * shorthand), the remaining terminal punctuation (. ! ?) is
 * reliable.  The boundary rule is:
 *
 *   char[i]  is in { '.', '!', '?' }
 *   char[i+1] is whitespace (or end-of-string / newline)
 *   char[i+2] is uppercase  (or end-of-string)
 *
 * Additionally, every newline in the input that is NOT inside a
 * sentence is treated as a hard boundary — this handles bullet
 * lists and already-split text files.
 *
 * Each extracted sentence is:
 *   - Stripped of leading and trailing whitespace.
 *   - Skipped if it contains only whitespace.
 * ============================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "sentence_splitter.h"
#include "preprocessing_types.h"

/* ---- helper: is character a sentence terminal? ---- */
static int is_terminal(char c) {
    return (c == '.' || c == '!' || c == '?');
}

/* ---- helper: strip leading and trailing whitespace in-place ---- */
static void strip_whitespace(char *s) {
    /* leading */
    int start = 0;
    while (s[start] && isspace((unsigned char)s[start])) start++;
    if (start > 0) memmove(s, s + start, strlen(s) - start + 1);

    /* trailing */
    int len = (int)strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }
}

/* ---- helper: check if string is blank ---- */
static int is_blank(const char *s) {
    while (*s) {
        if (!isspace((unsigned char)*s)) return 0;
        s++;
    }
    return 1;
}

/* ---- helper: add sentence to list ---- */
static int add_sentence(SentenceList *sl, const char *buf, int len) {
    if (sl->count >= MAX_SENTENCES) return PP_ERR_OVERFLOW;
    if (len <= 0 || len >= MAX_SENTENCE_LEN) return PP_ERR_OVERFLOW;

    strncpy(sl->data[sl->count], buf, len);
    sl->data[sl->count][len] = '\0';
    strip_whitespace(sl->data[sl->count]);

    if (!is_blank(sl->data[sl->count])) {
        sl->count++;
    }
    return PP_OK;
}

/* ================================================================
 * split_sentences()
 * ================================================================ */
int split_sentences(const char *text, SentenceList *out) {
    if (!text || !out) return PP_ERR_NULL;

    out->count = 0;
    memset(out->data, 0, sizeof(out->data));

    int   len      = (int)strlen(text);
    int   start    = 0;          /* start of current sentence */
    int   i        = 0;

    while (i < len) {

        char cur  = text[i];
        char next = (i + 1 < len) ? text[i + 1] : '\0';
        char nnxt = (i + 2 < len) ? text[i + 2] : '\0';

        /* ---- Hard boundary: newline acting as separator ---- */
        if (cur == '\n') {
            /* If we have content since last boundary, flush it */
            if (i > start) {
                add_sentence(out, text + start, i - start);
            }
            start = i + 1;
            i++;
            continue;
        }

        /* ---- Heuristic terminal punctuation boundary ---- */
        if (is_terminal(cur)) {
            /* Condition: next char is space (or end), char after is upper */
            int next_is_space = (next == '\0' || isspace((unsigned char)next));
            int after_is_upper = (nnxt == '\0' || isupper((unsigned char)nnxt));

            if (next_is_space && after_is_upper) {
                /* Include the terminal punctuation in current sentence */
                int seg_len = i - start + 1;
                add_sentence(out, text + start, seg_len);
                /* Skip the whitespace that follows */
                i++;
                while (i < len && isspace((unsigned char)text[i])) i++;
                start = i;
                continue;
            }
        }

        i++;
    }

    /* ---- Flush any remaining content ---- */
    if (start < len) {
        add_sentence(out, text + start, len - start);
    }

    return (out->count > 0) ? PP_OK : PP_ERR_EMPTY;
}

/* ================================================================
 * print_sentence_list()
 * ================================================================ */
void print_sentence_list(const SentenceList *sl) {
    if (!sl) { printf("(null SentenceList)\n"); return; }
    printf("SentenceList [%d sentence(s)]\n", sl->count);
    printf("%-4s  %s\n", "IDX", "SENTENCE");
    printf("%-4s  %s\n", "---", "--------");
    for (int i = 0; i < sl->count; i++) {
        printf("[%02d]  %s\n", i + 1, sl->data[i]);
    }
}
