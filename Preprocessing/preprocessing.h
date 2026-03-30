/* =============================================================
 * preprocessing.h
 * Public interface for the complete preprocessing pipeline.
 *
 * Other phases (thematic clustering, concept extraction …) should
 * #include this file and call run_preprocessing() to obtain a
 * clean SentenceList ready for downstream NLP.
 *
 * Pipeline order (fixed)
 * ----------------------
 *   1. expand_abbreviations()        — abbreviation_handler.c
 *   2. split_sentences()             — sentence_splitter.c
 *   3. remove_punctuation_from_list()— punctuation_removal.c
 *   4. correct_spelling_in_list()    — spelling_correction.c
 * ============================================================= */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "preprocessing_types.h"

/* ---------------------------------------------------------------
 * PreprocessingConfig
 *
 * Allows callers to toggle individual stages and supply optional
 * file paths.  Set a flag to 0 to skip that stage.
 * --------------------------------------------------------------- */
typedef struct {
    int  do_abbrev_expansion;   /* default: 1 */
    int  do_sentence_split;     /* default: 1 */
    int  do_punctuation_removal;/* default: 1 */
    int  do_spelling_correction;/* default: 1 */
    int  verbose;               /* print progress to stdout: 1/0 */
    const char *dict_filepath;  /* extra dictionary for spell-check;
                                   NULL = built-in words only       */
} PreprocessingConfig;

/* ---------------------------------------------------------------
 * default_config()
 * Returns a PreprocessingConfig with all stages enabled and
 * verbose off.  Start here and override what you need.
 * --------------------------------------------------------------- */
PreprocessingConfig default_config(void);

/* ---------------------------------------------------------------
 * run_preprocessing()
 *
 * Executes the full pipeline on 'raw_text' and fills 'out' with
 * the clean, sentence-split result.
 *
 * Parameters
 *   raw_text  : null-terminated raw document string
 *   out       : caller-allocated SentenceList (will be cleared)
 *   cfg       : pipeline configuration (pass NULL for defaults)
 *
 * Returns PP_OK on success, error code on failure.
 * --------------------------------------------------------------- */
int run_preprocessing(const char *raw_text,
                      SentenceList *out,
                      const PreprocessingConfig *cfg);

/* ---------------------------------------------------------------
 * run_preprocessing_from_file()
 *
 * Convenience wrapper: reads raw text from 'filepath', runs the
 * full pipeline, and fills 'out'.
 *
 * Returns PP_OK on success, PP_ERR_NULL on bad args, or -4 if
 * the file cannot be opened.
 * --------------------------------------------------------------- */
int run_preprocessing_from_file(const char *filepath,
                                SentenceList *out,
                                const PreprocessingConfig *cfg);

/* ---------------------------------------------------------------
 * preprocessing_error_str()
 * Returns a human-readable string for a PP_* error code.
 * --------------------------------------------------------------- */
const char *preprocessing_error_str(int error_code);

#endif /* PREPROCESSING_H */
