#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKENS 256
#define MAX_TOKEN_LEN 64

int tokenize_sentence(char *sentence, char tokens[][MAX_TOKEN_LEN]) {
    int token_count = 0;
    int i = 0;
    int len = strlen(sentence);

    while (i < len) {

        // Skip whitespace
        while (i < len && isspace(sentence[i])) i++;
        if (i >= len) break;

        int j = 0;
        char current_token[MAX_TOKEN_LEN];

        while (i < len && !isspace(sentence[i])) {
            char c = sentence[i];
            char next = (i + 1 < len) ? sentence[i + 1] : '\0';

            if (ispunct(c)) {
                if (c == '-' && j > 0 && isalpha(next)) {
                    current_token[j++] = c;
                } else if (c == '\'' && next == 's') {
                    current_token[j++] = c;
                } else {
                    i++;
                    break;
                }
            } else {
                current_token[j++] = c;
            }
            i++;
        }

        if (j > 0) {
            current_token[j] = '\0';
            strncpy(tokens[token_count], current_token, MAX_TOKEN_LEN);
            token_count++;
        }
    }

    return token_count;
}