#ifndef POS_TAGGER_H
#define POS_TAGGER_H

#define MAX_TOKENS 256
#define MAX_TOKEN_LEN 64
#define MAX_TAG_LEN 8

void tagger(char tokens[][MAX_TOKEN_LEN], int count, char tags[][MAX_TAG_LEN]);

#endif