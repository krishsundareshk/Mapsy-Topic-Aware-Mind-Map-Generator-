#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKENS 256
#define MAX_TOKEN_LEN 64
#define MAX_TAG_LEN 8

// ─── Known Word Lists ─────────────────────────────────────────────────────────

const char *determiners[]   = { "the", "a", "an", "this", "that", "these", "those", "each", "every", NULL };
const char *auxiliaries[]   = { "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", NULL };
const char *modals[]        = { "can", "could", "will", "would", "shall", "should", "may", "might", "must", NULL };
const char *prepositions[]  = { "in", "on", "at", "by", "for", "with", "of", "from", "into", "onto", "upon", "within", "without", "through", "between", "among", "during", "against", "across", "over", "under", "about", "along", "behind", "below", "beside", "beyond", "despite", "except", "near", NULL };
const char *conjunctions[]  = { "and", "or", "but", "nor", "yet", "so", NULL };
const char *subordinators[] = { "that", "which", "who", "whom", "whose", "when", "where", "because", "although", "while", "if", "unless", "since", "though", "whereas", "whether", NULL };
const char *pronouns[]      = { "it", "its", "they", "their", "them", "we", "our", "i", "my", "he", "his", "she", "her", "you", "your", NULL };

static const char *bare_verbs[] = {
    // cognitive / reporting
    "show", "shows", "suggest", "suggests", "indicate", "indicates",
    "reveal", "reveals", "confirm", "confirms", "demonstrate", "demonstrates",
    "find", "finds", "note", "notes", "argue", "argues", "claim", "claims",
    "propose", "proposes", "assume", "assumes", "conclude", "concludes",

    // physical / technical
    "use", "uses", "enable", "enables", "allow", "allows", "offer", "offers",
    "provide", "provides", "require", "requires", "involve", "involves",
    "include", "includes", "contain", "contains", "cause", "causes",
    "reduce", "reduces", "increase", "increases", "improve", "improves",
    "affect", "affects", "impact", "impacts", "support", "supports",
    "produce", "produces", "generate", "generates", "create", "creates",
    "convert", "converts",
    "capture", "captures",
    "make", "makes", "take", "takes", "give", "gives", "get", "gets",
    "help", "helps", "keep", "keeps",
    "rise", "rises",
    "grow", "grows", "remain", "remains", "represent", "represents",

    // domain specific to climate/energy text
    "emit", "emits",
    "tap", "taps", "warn", "warns", "harness", "harnesses", "mitigate",
    "mitigates", "transition", "transitions", "expand", "expands",
    "install", "installs", "adopt", "adopts", "deploy", "deploys",

    // base forms (appear after modals/auxiliaries)
    "learn", "train", "test", "build", "design",
    "extract", "predict", "classify", "cluster", "embed", "compute",
    "optimize", "converge", "update", "sample", "apply", "combine",
    "compare", "evaluate", "measure", "define", "fit",

    NULL
};


// ─── Helpers ──────────────────────────────────────────────────────────────────

void to_lower(const char *src, char *dst) {
    int i = 0;
    while (src[i]) { dst[i] = tolower((unsigned char)src[i]); i++; }
    dst[i] = '\0';
}

int in_list(const char *word, const char **list) {
    for (int i = 0; list[i] != NULL; i++)
        if (strcmp(word, list[i]) == 0) return 1;
    return 0;
}

int has_suffix(const char *word, const char *suffix) {
    int wlen = strlen(word), slen = strlen(suffix);
    if (wlen <= slen) return 0;
    return strcmp(word + wlen - slen, suffix) == 0;
}

int all_digits(const char *word) {
    for (int i = 0; word[i]; i++)
        if (!isdigit((unsigned char)word[i])) return 0;
    return 1;
}

int all_caps(const char *word) {
    for (int i = 0; word[i]; i++)
        if (!isupper((unsigned char)word[i])) return 0;
    return 1;
}

// ─── Pass 1: Tag independently ────────────────────────────────────────────────

void pass1(char tokens[][MAX_TOKEN_LEN], int count, char tags[][MAX_TAG_LEN]) {
    for (int i = 0; i < count; i++) {
        char *token = tokens[i];
        char *tag   = tags[i];
        char lower[MAX_TOKEN_LEN];
        to_lower(token, lower);

        if (all_digits(lower))                     { strcpy(tag, "NUM");  continue; }
        if (strcmp(lower, "to") == 0)              { strcpy(tag, "TO");   continue; }
        if (in_list(lower, determiners))           { strcpy(tag, "DET");  continue; }
        if (in_list(lower, auxiliaries))           { strcpy(tag, "AUX");  continue; }
        if (in_list(lower, modals))                { strcpy(tag, "MOD");  continue; }
        if (in_list(lower, prepositions))          { strcpy(tag, "PREP"); continue; }
        if (in_list(lower, conjunctions))          { strcpy(tag, "CONJ"); continue; }
        if (in_list(lower, subordinators))         { strcpy(tag, "SUB");  continue; }
        if (in_list(lower, pronouns))              { strcpy(tag, "PRON"); continue; }
        if (in_list(lower, bare_verbs))            { strcpy(tag, "VERB"); continue; }
        if (all_caps(token) && strlen(token) > 1)  { strcpy(tag, "PROPN"); continue; }

        if (has_suffix(lower, "isation") || has_suffix(lower, "ization") ||
            has_suffix(lower, "tion")    || has_suffix(lower, "sion")    ||
            has_suffix(lower, "ment")    || has_suffix(lower, "ness")    ||
            has_suffix(lower, "ity")     || has_suffix(lower, "ance")    ||
            has_suffix(lower, "ence"))             { strcpy(tag, "NOUN"); continue; }

        if (has_suffix(lower, "ing"))              { strcpy(tag, "VERB"); continue; }
        if (has_suffix(lower, "ed"))               { strcpy(tag, "VERB"); continue; }

        if (has_suffix(lower, "ise")  || has_suffix(lower, "ises") ||
            has_suffix(lower, "ize")  || has_suffix(lower, "izes") ||
            has_suffix(lower, "ate")  || has_suffix(lower, "ates") ||
            has_suffix(lower, "fy")   || has_suffix(lower, "fies") ||
            has_suffix(lower, "ges")  ||   // converges, emerges
            has_suffix(lower, "ves")  ||   // solves, involves
            has_suffix(lower, "ses")  ||   // processes
            has_suffix(lower, "zes")  ||   // analyzes
            has_suffix(lower, "rms")  ||   // performs, transforms
            has_suffix(lower, "nds")  ||   // extends, depends
            has_suffix(lower, "rns"))      // returns, learns
            { strcpy(tag, "VERB"); continue; }

        if (has_suffix(lower, "ly"))               { strcpy(tag, "ADV");  continue; }

        if (has_suffix(lower, "al")   || has_suffix(lower, "ic")   ||
            has_suffix(lower, "ive")  || has_suffix(lower, "ous")  ||
            has_suffix(lower, "able") || has_suffix(lower, "ible")) { strcpy(tag, "ADJ"); continue; }

        if (i > 0 && isupper((unsigned char)token[0])) { strcpy(tag, "PROPN"); continue; }

        strcpy(tag, "NOUN");
    }
}

// ─── Pass 2: Correct using context ───────────────────────────────────────────

void pass2(char tokens[][MAX_TOKEN_LEN], int count, char tags[][MAX_TAG_LEN]) {
    for (int i = 0; i < count; i++) {
        char *cur_tag   = tags[i];
        char cur_lower[MAX_TOKEN_LEN];
        to_lower(tokens[i], cur_lower);

        char *left_tag  = (i > 0)           ? tags[i-1] : NULL;
        char *right_tag = (i < count - 1)   ? tags[i+1] : NULL;

        if (left_tag && strcmp(left_tag, "TO")   == 0 && strcmp(cur_tag, "NOUN") == 0) { strcpy(cur_tag, "VERB"); continue; }
        if (left_tag && strcmp(left_tag, "MOD")  == 0 && strcmp(cur_tag, "NOUN") == 0) { strcpy(cur_tag, "VERB"); continue; }
        if (left_tag && strcmp(left_tag, "AUX")  == 0 && strcmp(cur_tag, "NOUN") == 0) { strcpy(cur_tag, "VERB"); continue; }

        if (left_tag && strcmp(cur_tag, "VERB") == 0 && has_suffix(cur_lower, "ing")) {
            if (strcmp(left_tag, "DET")  == 0 ||
                strcmp(left_tag, "PREP") == 0 ||
                strcmp(left_tag, "PRON") == 0) { strcpy(cur_tag, "NOUN"); continue; }
        }
        if (right_tag && strcmp(cur_tag, "VERB") == 0 && has_suffix(cur_lower, "ing")) {
            if (strcmp(right_tag, "NOUN") == 0) { strcpy(cur_tag, "NOUN"); continue; }
        }
        if (left_tag && strcmp(cur_tag, "VERB") == 0 && has_suffix(cur_lower, "ed")) {
            if (strcmp(left_tag, "DET") == 0) { strcpy(cur_tag, "ADJ"); continue; }
        }
        if (right_tag && strcmp(cur_tag, "VERB") == 0 && has_suffix(cur_lower, "ed")) {
            if (strcmp(right_tag, "NOUN") == 0) { strcpy(cur_tag, "ADJ"); continue; }
        }
        if (left_tag && strcmp(left_tag, "CONJ") == 0) {
            char *left_left_tag = (i > 1) ? tags[i-2] : NULL;
            if (left_left_tag && strcmp(cur_tag, "NOUN") == 0) { strcpy(cur_tag, left_left_tag); continue; }
        }
        if (left_tag && strcmp(left_tag, "DET") == 0 && strcmp(cur_tag, "ADJ") == 0) {
            if (right_tag == NULL || strcmp(right_tag, "PREP") == 0) {
                strcpy(cur_tag, "NOUN"); continue;
            }
        } 
    }
}

// ─── Public function ──────────────────────────────────────────────────────────

void tagger(char tokens[][MAX_TOKEN_LEN], int count, char tags[][MAX_TAG_LEN]) {
    pass1(tokens, count, tags);
    pass2(tokens, count, tags);
}
