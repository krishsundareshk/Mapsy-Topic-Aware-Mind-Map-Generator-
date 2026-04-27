#ifndef CHUNKER_H
#define CHUNKER_H

#define MAX_TOKENS 256
#define MAX_TOKEN_LEN 64
#define MAX_TAG_LEN 8
#define MAX_CHUNKS 128
#define MAX_CHUNK_TOKENS 16

int chunk(
    char tokens[][MAX_TOKEN_LEN],
    char tags[][MAX_TAG_LEN],
    int count,
    char chunk_labels[][MAX_TAG_LEN],
    char chunk_tokens[][MAX_CHUNK_TOKENS][MAX_TOKEN_LEN],
    int chunk_sizes[]
);

#endif