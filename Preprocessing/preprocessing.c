/* =============================================================
 * preprocessing.c
 * Orchestrates the four preprocessing stages in order.
 * ============================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "preprocessing.h"
#include "abbreviation_handler.h"
#include "sentence_splitter.h"
#include "punctuation_removal.h"
#include <ctype.h>
#include "spelling_correction.h"
#include "preprocessing_types.h"

/* ================================================================
 * default_config()
 * ================================================================ */
PreprocessingConfig default_config(void) {
    PreprocessingConfig cfg;
    cfg.do_abbrev_expansion    = 1;
    cfg.do_sentence_split      = 1;
    cfg.do_punctuation_removal = 1;
    cfg.do_spelling_correction = 1;
    cfg.verbose                = 0;
    cfg.dict_filepath          = NULL;
    return cfg;
}

/* ================================================================
 * preprocessing_error_str()
 * ================================================================ */
const char *preprocessing_error_str(int code) {
    switch (code) {
        case PP_OK:           return "Success";
        case PP_ERR_NULL:     return "Null pointer argument";
        case PP_ERR_OVERFLOW: return "Buffer overflow / too many sentences";
        case PP_ERR_EMPTY:    return "Result is empty";
        default:              return "Unknown error";
    }
}


/* ================================================================
 * capitalize_sentences() — internal helper
 * Ensures the first alphabetic character of every sentence is
 * uppercase.  Handles cases where abbreviation expansion lowered
 * the initial word (e.g. "Govts." -> "governments").
 * ================================================================ */
static void capitalize_sentences(SentenceList *sl) {
    for (int i = 0; i < sl->count; i++) {
        char *s = sl->data[i];
        for (int j = 0; s[j]; j++) {
            if (isalpha((unsigned char)s[j])) {
                s[j] = (char)toupper((unsigned char)s[j]);
                break;
            }
        }
    }
}

/* ================================================================
 * run_preprocessing()
 * ================================================================ */
int run_preprocessing(const char *raw_text,
                      SentenceList *out,
                      const PreprocessingConfig *cfg) {

    if (!raw_text || !out) return PP_ERR_NULL;

    /* Resolve configuration */
    PreprocessingConfig conf = cfg ? *cfg : default_config();

    /* Clear output */
    memset(out, 0, sizeof(SentenceList));

    int ret = PP_OK;

    /* ----------------------------------------------------------
     * Stage 1 — Abbreviation Expansion
     * Operates on the raw text string before splitting.
     * ---------------------------------------------------------- */
    static char expanded_text[MAX_TEXT_LEN];
    const char *text_to_split = raw_text;

    if (conf.do_abbrev_expansion) {
        if (conf.verbose)
            printf("[preprocessing] Stage 1: Abbreviation Expansion...\n");

        ret = expand_abbreviations(raw_text, expanded_text);
        if (ret != PP_OK) {
            fprintf(stderr, "[preprocessing] Abbreviation expansion failed: %s\n",
                    preprocessing_error_str(ret));
            /* Fall back to raw text */
            strncpy(expanded_text, raw_text, MAX_TEXT_LEN - 1);
            expanded_text[MAX_TEXT_LEN - 1] = '\0';
        }
        text_to_split = expanded_text;

        if (conf.verbose)
            printf("[preprocessing] Stage 1 done.\n");
    }

    /* ----------------------------------------------------------
     * Stage 2 — Sentence Splitting
     * ---------------------------------------------------------- */
    if (conf.do_sentence_split) {
        if (conf.verbose)
            printf("[preprocessing] Stage 2: Sentence Splitting...\n");

        ret = split_sentences(text_to_split, out);
        if (ret != PP_OK && ret != PP_ERR_EMPTY) {
            fprintf(stderr, "[preprocessing] Sentence splitting failed: %s\n",
                    preprocessing_error_str(ret));
            return ret;
        }
        if (conf.verbose)
            printf("[preprocessing] Stage 2 done. %d sentence(s) found.\n",
                   out->count);
    } else {
        /* No splitting: treat entire text as one sentence */
        strncpy(out->data[0], text_to_split, MAX_SENTENCE_LEN - 1);
        out->data[0][MAX_SENTENCE_LEN - 1] = '\0';
        out->count = 1;
    }

    /* ----------------------------------------------------------
     * Stage 3 — Punctuation Removal
     * ---------------------------------------------------------- */
    if (conf.do_punctuation_removal) {
        if (conf.verbose)
            printf("[preprocessing] Stage 3: Punctuation Removal...\n");

        ret = remove_punctuation_from_list(out);
        if (ret != PP_OK)
            fprintf(stderr, "[preprocessing] Punctuation removal error: %s\n",
                    preprocessing_error_str(ret));

        if (conf.verbose)
            printf("[preprocessing] Stage 3 done.\n");
    }

    /* ----------------------------------------------------------
     * Stage 4 — Spelling Correction
     * ---------------------------------------------------------- */
    if (conf.do_spelling_correction) {
        if (conf.verbose)
            printf("[preprocessing] Stage 4: Spelling Correction...\n");

        /* Load optional extra dictionary */
        sc_load_dictionary(conf.dict_filepath);

        ret = correct_spelling_in_list(out);
        if (ret != PP_OK)
            fprintf(stderr, "[preprocessing] Spelling correction error: %s\n",
                    preprocessing_error_str(ret));

        if (conf.verbose)
            printf("[preprocessing] Stage 4 done.\n");
    }

    /* Capitalise first character of every sentence */
    capitalize_sentences(out);

    if (conf.verbose) {
        printf("[preprocessing] Pipeline complete. "
               "%d clean sentence(s) produced.\n", out->count);
    }

    return PP_OK;
}

/* ================================================================
 * run_preprocessing_from_file()
 * ================================================================ */
int run_preprocessing_from_file(const char *filepath,
                                SentenceList *out,
                                const PreprocessingConfig *cfg) {
    if (!filepath || !out) return PP_ERR_NULL;

    FILE *f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "[preprocessing] Cannot open file: %s\n", filepath);
        return -4;
    }

    /* Read entire file into a buffer */
    static char file_buf[MAX_TEXT_LEN];
    size_t n = fread(file_buf, 1, MAX_TEXT_LEN - 1, f);
    fclose(f);
    file_buf[n] = '\0';

    return run_preprocessing(file_buf, out, cfg);
}

/* ================================================================
 * main() — standalone driver for testing
 *
 * Usage:
 *   ./preprocessing <input_file> [dictionary_file]
 *
 * Compile:
 *   gcc -Wall -o preprocessing \
 *       preprocessing.c abbreviation_handler.c sentence_splitter.c \
 *       punctuation_removal.c spelling_correction.c
 * ================================================================ */
#ifdef PREPROCESSING_MAIN

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [dict_file]\n", argv[0]);
        return 1;
    }

    SentenceList result;
    PreprocessingConfig cfg = default_config();
    cfg.verbose       = 1;
    cfg.dict_filepath = (argc >= 3) ? argv[2] : NULL;

    int ret = run_preprocessing_from_file(argv[1], &result, &cfg);

    if (ret != PP_OK) {
        fprintf(stderr, "Preprocessing failed: %s\n",
                preprocessing_error_str(ret));
        return 1;
    }

    printf("\n========== FINAL OUTPUT ==========\n");
    print_sentence_list(&result);

    return 0;
}

#endif /* PREPROCESSING_MAIN */
