/* =============================================================
 * preprocessing_types.h
 * Shared data structures and constants for the preprocessing
 * pipeline. Include this in every preprocessing module.
 * ============================================================= */

#ifndef PREPROCESSING_TYPES_H
#define PREPROCESSING_TYPES_H

/* ---------- tuneable limits ---------- */
#define MAX_SENTENCES      150
#define MAX_SENTENCE_LEN   1024
#define MAX_TEXT_LEN       32768   /* max raw document size   */
#define MAX_WORD_LEN       128     /* max single-word length  */
#define MAX_WORDS_IN_SENT  300     /* max words per sentence  */

/* ---------- return / status codes ---------- */
#define PP_OK              0
#define PP_ERR_NULL        -1
#define PP_ERR_OVERFLOW    -2
#define PP_ERR_EMPTY       -3

/* ---------- primary output container ----------
 * Every stage reads from and/or writes to a SentenceList.
 * 'count'  : number of valid sentences stored.
 * 'data'   : null-terminated C strings, one per sentence.
 * ------------------------------------------------ */
typedef struct {
    char data[MAX_SENTENCES][MAX_SENTENCE_LEN];
    int  count;
} SentenceList;

#endif /* PREPROCESSING_TYPES_H */
