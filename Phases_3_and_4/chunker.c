#include "chunker.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static int is_tag(char *tag, const char *expected) {
    return strcmp(tag, expected) == 0;
}

static int has_suffix(const char *word, const char *suffix) {
    int wlen = strlen(word), slen = strlen(suffix);
    if (wlen <= slen) return 0;
    return strcmp(word + wlen - slen, suffix) == 0;
}

// ─── Pattern matchers ─────────────────────────────────────────────────────────

// Tries to match an NP starting at position i
// Returns number of tokens consumed, 0 if no match
static int match_np(
    char tokens[][MAX_TOKEN_LEN],
    char tags[][MAX_TAG_LEN],
    int count, int i,
    char chunk_tok[][MAX_TOKEN_LEN],
    int *chunk_size)
{
    int start = i;
    *chunk_size = 0;

    // optional DET
    if (i < count && is_tag(tags[i], "DET")) {
        strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);
    }

    // zero or more ADJ
    while (i < count && is_tag(tags[i], "ADJ")) {
        strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);
    }

    // one or more NOUN or PROPN — required
    if (i >= count || (!is_tag(tags[i], "NOUN") && !is_tag(tags[i], "PROPN"))) {
        return 0; // no match
    }
    while (i < count && (is_tag(tags[i], "NOUN") || is_tag(tags[i], "PROPN"))) {
        strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);
    }

    return i - start; // tokens consumed
}

// Tries to match a VP starting at position i
// Returns number of tokens consumed, 0 if no match
static int match_vp(
    char tokens[][MAX_TOKEN_LEN],
    char tags[][MAX_TAG_LEN],
    int count, int i,
    char chunk_tok[][MAX_TOKEN_LEN],
    int *chunk_size)
{
    int start = i;
    *chunk_size = 0;

    // zero or more AUX or MOD
    while (i < count && (is_tag(tags[i], "AUX") || is_tag(tags[i], "MOD"))) {
        strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);
    }

    // optional ADV
    if (i < count && is_tag(tags[i], "ADV")) {
        strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);
    }

    // one or more VERB — required
    if (i >= count || !is_tag(tags[i], "VERB")) {
        return 0; // no match
    }
    while (i < count && is_tag(tags[i], "VERB")) {
        strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);
    }

    return i - start;
}

// Tries to match a PP starting at position i
// Returns number of tokens consumed, 0 if no match
static int match_pp(
    char tokens[][MAX_TOKEN_LEN],
    char tags[][MAX_TAG_LEN],
    int count, int i,
    char chunk_tok[][MAX_TOKEN_LEN],
    int *chunk_size)
{
    int start = i;
    *chunk_size = 0;

    // must start with PREP
    if (i >= count || !is_tag(tags[i], "PREP")) return 0;
    strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);

    // check for gerund — VERB(-ing) treated as NOUN
    if (i < count && is_tag(tags[i], "VERB") && has_suffix(tokens[i], "ing")) {
        strncpy(chunk_tok[(*chunk_size)++], tokens[i++], MAX_TOKEN_LEN);
        return i - start;
    }

    // otherwise expect a full NP
    char np_tokens[MAX_CHUNK_TOKENS][MAX_TOKEN_LEN];
    int np_size = 0;
    int np_consumed = match_np(tokens, tags, count, i, np_tokens, &np_size);

    if (np_consumed == 0) return 0; // no NP after PREP, no match

    for (int j = 0; j < np_size; j++) {
        strncpy(chunk_tok[(*chunk_size)++], np_tokens[j], MAX_TOKEN_LEN);
    }
    i += np_consumed;

    return i - start;
}

// ─── Main chunker ─────────────────────────────────────────────────────────────

int chunk(
    char tokens[][MAX_TOKEN_LEN],
    char tags[][MAX_TAG_LEN],
    int count,
    char chunk_labels[][MAX_TAG_LEN],
    char chunk_tokens[][MAX_CHUNK_TOKENS][MAX_TOKEN_LEN],
    int chunk_sizes[])
{
    int i = 0;
    int chunk_count = 0;

    while (i < count) {
        char tmp_tokens[MAX_CHUNK_TOKENS][MAX_TOKEN_LEN];
        int tmp_size = 0;
        int consumed = 0;

        // Try NP
        consumed = match_np(tokens, tags, count, i, tmp_tokens, &tmp_size);
        if (consumed > 0) {
            strcpy(chunk_labels[chunk_count], "NP");
            for (int j = 0; j < tmp_size; j++)
                strncpy(chunk_tokens[chunk_count][j], tmp_tokens[j], MAX_TOKEN_LEN);
            chunk_sizes[chunk_count] = tmp_size;
            chunk_count++;
            i += consumed;
            continue;
        }

        // Try VP
        consumed = match_vp(tokens, tags, count, i, tmp_tokens, &tmp_size);
        if (consumed > 0) {
            strcpy(chunk_labels[chunk_count], "VP");
            for (int j = 0; j < tmp_size; j++)
                strncpy(chunk_tokens[chunk_count][j], tmp_tokens[j], MAX_TOKEN_LEN);
            chunk_sizes[chunk_count] = tmp_size;
            chunk_count++;
            i += consumed;
            continue;
        }

        // Try PP
        consumed = match_pp(tokens, tags, count, i, tmp_tokens, &tmp_size);
        if (consumed > 0) {
            strcpy(chunk_labels[chunk_count], "PP");
            for (int j = 0; j < tmp_size; j++)
                strncpy(chunk_tokens[chunk_count][j], tmp_tokens[j], MAX_TOKEN_LEN);
            chunk_sizes[chunk_count] = tmp_size;
            chunk_count++;
            i += consumed;
            continue;
        }

        // No match — OTHER
        strcpy(chunk_labels[chunk_count], "OTHER");
        strncpy(chunk_tokens[chunk_count][0], tokens[i], MAX_TOKEN_LEN);
        chunk_sizes[chunk_count] = 1;
        chunk_count++;
        i++;
    }

    return chunk_count;
}