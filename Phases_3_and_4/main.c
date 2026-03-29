#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "tokenizer.h"
#include "pos_tagger.h"
#include "chunker.h"

#define MAX_SENTENCE_LEN 512
#define MAX_SENTENCES 64

// ─── Extract string value from a JSON line ────────────────────────────────────

int extract_string_value(const char *line, const char *key, char *output) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);

    char *start = strstr(line, search);
    if (!start) return 0;

    start += strlen(search);

    // skip whitespace
    while (*start == ' ') start++;

    // must start with quote
    if (*start != '"') return 0;
    start++;

    char *end = strrchr(start, '"');
    if (!end || end == start - 1) return 0;

    strncpy(output, start, end - start);
    output[end - start] = '\0';
    return 1;
}

// ─── Extract integer value from a JSON line ───────────────────────────────────

int extract_int_value(const char *line, const char *key, int *output) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);

    char *start = strstr(line, search);
    if (!start) return 0;

    start += strlen(search);
    while (*start == ' ') start++;

    if (!isdigit(*start)) return 0;
    *output = atoi(start);
    return 1;
}

// ─── Write chunk list to file ─────────────────────────────────────────────────

void write_sentence_chunks(
    FILE *f,
    int sentence_id,
    double score,
    char *text,
    char chunk_labels[][MAX_TAG_LEN],
    char chunk_tokens[][MAX_CHUNK_TOKENS][MAX_TOKEN_LEN],
    int chunk_sizes[],
    int chunk_count,
    int is_last)
{
    fprintf(f, "      {\n");
    fprintf(f, "        \"id\": %d,\n", sentence_id);
    fprintf(f, "        \"score\": %.3f,\n", score);
    fprintf(f, "        \"text\": \"%s\",\n", text);
    fprintf(f, "        \"chunks\": [\n");

    for (int i = 0; i < chunk_count; i++) {
        fprintf(f, "          {\"label\": \"%s\", \"tokens\": [", chunk_labels[i]);
        for (int j = 0; j < chunk_sizes[i]; j++) {
            fprintf(f, "\"%s\"", chunk_tokens[i][j]);
            if (j < chunk_sizes[i] - 1) fprintf(f, ", ");
        }
        fprintf(f, "]}");
        if (i < chunk_count - 1) fprintf(f, ",");
        fprintf(f, "\n");
    }

    fprintf(f, "        ]\n");
    fprintf(f, "      }");
    if (!is_last) fprintf(f, ",");
    fprintf(f, "\n");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    FILE *in = fopen("input.json", "r");
    if (!in) { printf("Error: could not open input.json\n"); return 1; }

    FILE *out = fopen("chunks.json", "w");
    if (!out) { printf("Error: could not open chunks.json\n"); return 1; }

    // buffers
    char line[1024];
    char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    char tags[MAX_TOKENS][MAX_TAG_LEN];
    char chunk_labels[MAX_CHUNKS][MAX_TAG_LEN];
    char chunk_tokens[MAX_CHUNKS][MAX_CHUNK_TOKENS][MAX_TOKEN_LEN];
    int  chunk_sizes[MAX_CHUNKS];

    // sentence buffer per topic
    typedef struct {
        int id;
        double score;
        char text[MAX_SENTENCE_LEN];
    } SentenceEntry;

    SentenceEntry sentences[MAX_SENTENCES];
    int sentence_count = 0;
    int current_topic_id = -1;
    int in_sentence = 0;
    int current_id = -1;
    double current_score = 0.0;
    char current_text[MAX_SENTENCE_LEN] = {0};

    // topic buffer — we collect all topics then write
    // simple approach: write as we parse, flush per topic

    fprintf(out, "{\n  \"topics\": [\n");

    int first_topic = 1;
    int topic_open = 0;

    while (fgets(line, sizeof(line), in)) {
        int val_int;
        char val_str[MAX_SENTENCE_LEN];

        // detect topic_id
        if (extract_int_value(line, "topic_id", &val_int)) {
            // close previous topic if open
            if (topic_open) {
                // write all buffered sentences for previous topic
                for (int s = 0; s < sentence_count; s++) {
                    int token_count = tokenize_sentence(sentences[s].text, tokens);
                    tagger(tokens, token_count, tags);
                    int chunk_count = chunk(
                        tokens, tags, token_count,
                        chunk_labels, chunk_tokens, chunk_sizes
                    );
                    write_sentence_chunks(
                        out,
                        sentences[s].id,
                        sentences[s].score,
                        sentences[s].text,
                        chunk_labels, chunk_tokens, chunk_sizes,
                        chunk_count,
                        s == sentence_count - 1
                    );
                }
                fprintf(out, "    ]\n    }");
                sentence_count = 0;
            }

            if (!first_topic) fprintf(out, ",\n");
            fprintf(out, "    {\n      \"topic_id\": %d,\n      \"sentences\": [\n", val_int);
            current_topic_id = val_int;
            first_topic = 0;
            topic_open = 1;
            continue;
        }

        // detect sentence id
        if (extract_int_value(line, "id", &val_int)) {
            current_id = val_int;
            continue;
        }

        // detect score
        if (strstr(line, "\"score\":")) {
            char *start = strstr(line, "\"score\":");
            start += strlen("\"score\":");
            while (*start == ' ') start++;
            current_score = atof(start);
            continue;
        }

        // detect text
        if (extract_string_value(line, "text", val_str)) {
            strncpy(current_text, val_str, MAX_SENTENCE_LEN);

            // save sentence to buffer
            sentences[sentence_count].id = current_id;
            sentences[sentence_count].score = current_score;
            strncpy(sentences[sentence_count].text, current_text, MAX_SENTENCE_LEN);
            sentence_count++;

            current_id = -1;
            current_score = 0.0;
            current_text[0] = '\0';
            continue;
        }
    }

    // flush last topic
    if (topic_open) {
        for (int s = 0; s < sentence_count; s++) {
            int token_count = tokenize_sentence(sentences[s].text, tokens);
            tagger(tokens, token_count, tags);
            int chunk_count = chunk(
                tokens, tags, token_count,
                chunk_labels, chunk_tokens, chunk_sizes
            );
            write_sentence_chunks(
                out,
                sentences[s].id,
                sentences[s].score,
                sentences[s].text,
                chunk_labels, chunk_tokens, chunk_sizes,
                chunk_count,
                s == sentence_count - 1
            );
        }
        fprintf(out, "    ]\n    }\n");
    }

    fprintf(out, "  ]\n}\n");

    fclose(in);
    fclose(out);

    printf("Done. Written to chunks.json\n");
    return 0;
}