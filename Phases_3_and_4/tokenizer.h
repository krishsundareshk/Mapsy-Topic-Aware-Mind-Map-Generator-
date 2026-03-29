#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_TOKENS 256
#define MAX_TOKEN_LEN 64

int tokenize_sentence(char *sentence, char tokens[][MAX_TOKEN_LEN]);

#endif