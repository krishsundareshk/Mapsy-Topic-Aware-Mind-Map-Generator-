/* =============================================================
 * abbreviation_handler.c
 *
 * Strategy
 * --------
 * 1. Walk the input character-by-character, collecting tokens
 *    (maximal runs of non-whitespace characters).
 * 2. For each token, do a case-insensitive lookup in the
 *    abbreviation table.
 * 3. If a match is found, write the expanded form; otherwise
 *    copy the token verbatim.
 * 4. Whitespace between tokens is preserved as-is.
 *
 * The table covers titles, Latin phrases, units, domain-specific
 * shorthand relevant to the sample corpus (climate / energy /
 * electric vehicles) and other common English abbreviations.
 * ============================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "abbreviation_handler.h"
#include "preprocessing_types.h"

/* ---- internal abbreviation table ---- */
typedef struct {
    const char *abbr;       /* abbreviation exactly as it may appear   */
    const char *expansion;  /* full-form replacement                   */
} AbbrevEntry;

static const AbbrevEntry ABBREV_TABLE[] = {
    /* ----- titles ----- */
    { "Dr.",    "Doctor"       },
    { "Mr.",    "Mister"       },
    { "Mrs.",   "Missus"       },
    { "Ms.",    "Miss"         },
    { "Prof.",  "Professor"    },
    { "Sr.",    "Senior"       },
    { "Jr.",    "Junior"       },

    /* ----- Latin / academic ----- */
    { "i.e.",   "that is"      },
    { "e.g.",   "for example"  },
    { "etc.",   "and so on"    },
    { "vs.",    "versus"       },
    { "et al.", "and others"   },
    { "viz.",   "namely"       },
    { "cf.",    "compare"      },
    { "ibid.",  "in the same place" },

    /* ----- common English ----- */
    { "approx.",  "approximately"  },
    { "incl.",    "including"      },
    { "excl.",    "excluding"      },
    { "max.",     "maximum"        },
    { "min.",     "minimum"        },
    { "avg.",     "average"        },
    { "est.",     "estimated"      },
    { "dept.",    "department"     },
    { "govt.",    "government"     },
    { "govts.",   "governments"    },
    { "org.",     "organisation"   },
    { "corp.",    "corporation"    },
    { "intl.",    "international"  },
    { "natl.",    "national"       },
    { "no.",      "number"         },
    { "fig.",     "figure"         },
    { "sec.",     "section"        },
    { "vol.",     "volume"         },
    { "yr.",      "year"           },
    { "yrs.",     "years"          },
    { "km.",      "kilometres"     },
    { "km",       "kilometres"     },
    { "mph",      "miles per hour" },
    { "kph",      "kilometres per hour" },
    { "pct.",     "percent"        },
    { "pct",      "percent"        },

    /* ----- domain: climate / energy / transport ----- */
    { "CO2",      "carbon dioxide"           },
    { "CH4",      "methane"                  },
    { "GHG",      "greenhouse gas"           },
    { "GHGs",     "greenhouse gases"         },
    { "IPCC",     "Intergovernmental Panel on Climate Change" },
    { "EV",       "electric vehicle"         },
    { "EVs",      "electric vehicles"        },
    { "ICE",      "internal combustion engine" },
    { "kWh",      "kilowatt-hours"           },
    { "MW",       "megawatts"                },
    { "GW",       "gigawatts"                },
    { "PV",       "photovoltaic"             },

    /* sentinel — must be last */
    { NULL, NULL }
};

/* ---- helper: case-insensitive token comparison ---- */
/* Returns 1 if the abbreviation string is ALL uppercase letters only
 * (e.g. "ICE", "IPCC", "EV") — used to decide match strategy.   */
static int is_all_upper_abbr(const char *abbr) {
    for (; *abbr; abbr++) {
        if (!isupper((unsigned char)*abbr)) return 0;
    }
    return 1;
}

static int token_match(const char *token, const char *abbr) {
    /* Strategy:
     * - Pure uppercase abbreviations (ICE, IPCC, EV, GHG …):
     *   CASE-SENSITIVE — prevents lowercase "ice" from matching.
     * - All other abbreviations (govts., approx., i.e. …):
     *   CASE-INSENSITIVE — handles sentence-initial capitalisation,
     *   e.g. "Govts." matches "govts.".                           */
    if (is_all_upper_abbr(abbr))
        return (strcmp(token, abbr) == 0);

    /* case-insensitive compare */
    size_t alen = strlen(abbr);
    size_t tlen = strlen(token);
    if (tlen != alen) return 0;
    for (size_t i = 0; i < alen; i++) {
        if (tolower((unsigned char)token[i]) !=
            tolower((unsigned char)abbr[i])) return 0;
    }
    return 1;
}

/* ---- helper: look up a token in the table ---- */
static const char *lookup_expansion(const char *token) {
    for (int i = 0; ABBREV_TABLE[i].abbr != NULL; i++) {
        if (token_match(token, ABBREV_TABLE[i].abbr))
            return ABBREV_TABLE[i].expansion;
    }
    return NULL;
}

/* ================================================================
 * expand_abbreviations()
 * ================================================================ */
int expand_abbreviations(const char *input, char *output) {
    if (!input || !output) return PP_ERR_NULL;

    const char *src  = input;
    char       *dst  = output;
    char       *end  = output + MAX_TEXT_LEN - 1;
    char        token[MAX_WORD_LEN];

    while (*src != '\0') {

        /* --- copy leading whitespace / newlines verbatim --- */
        if (isspace((unsigned char)*src)) {
            if (dst >= end) return PP_ERR_OVERFLOW;
            *dst++ = *src++;
            continue;
        }

        /* --- collect one non-whitespace token --- */
        int tlen = 0;
        const char *tok_start = src;
        while (*src && !isspace((unsigned char)*src)) {
            if (tlen < MAX_WORD_LEN - 1)
                token[tlen++] = *src;
            src++;
        }
        token[tlen] = '\0';

        /* --- try to expand --- */
        const char *expansion = lookup_expansion(token);
        const char *write_str = expansion ? expansion : token;

        size_t wlen = strlen(write_str);
        if (dst + wlen >= end) return PP_ERR_OVERFLOW;
        memcpy(dst, write_str, wlen);
        dst += wlen;

        (void)tok_start; /* suppress unused-variable warning */
    }

    *dst = '\0';
    return PP_OK;
}

/* ================================================================
 * list_abbreviations()
 * ================================================================ */
void list_abbreviations(void) {
    printf("%-20s  ->  %s\n", "ABBREVIATION", "EXPANSION");
    printf("%-20s      %s\n", "------------", "---------");
    for (int i = 0; ABBREV_TABLE[i].abbr != NULL; i++) {
        printf("%-20s  ->  %s\n",
               ABBREV_TABLE[i].abbr,
               ABBREV_TABLE[i].expansion);
    }
}
