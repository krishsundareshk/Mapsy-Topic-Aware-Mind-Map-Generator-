/* =============================================================
 * abbreviation_handler.h
 * Expands common abbreviations in a raw text block BEFORE
 * sentence splitting so that periods inside abbreviations
 * (e.g. "approx.") never cause false sentence boundaries.
 * ============================================================= */

#ifndef ABBREVIATION_HANDLER_H
#define ABBREVIATION_HANDLER_H

#include "preprocessing_types.h"

/* ---------------------------------------------------------------
 * expand_abbreviations()
 *
 * Reads 'input' (a raw document string) and writes an expanded
 * version into 'output' (caller-allocated, size MAX_TEXT_LEN).
 *
 * All recognised abbreviations are replaced with their full
 * forms.  Unknown tokens are copied verbatim.
 *
 * Returns: PP_OK on success, PP_ERR_NULL / PP_ERR_OVERFLOW on
 *          failure.
 * --------------------------------------------------------------- */
int expand_abbreviations(const char *input, char *output);

/* ---------------------------------------------------------------
 * list_abbreviations()
 *
 * Prints the built-in abbreviation table to stdout.
 * Useful for debugging / documentation.
 * --------------------------------------------------------------- */
void list_abbreviations(void);

#endif /* ABBREVIATION_HANDLER_H */
