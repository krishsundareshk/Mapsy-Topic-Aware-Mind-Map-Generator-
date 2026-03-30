/* =============================================================
 * spelling_correction.c
 *
 * Strategy
 * --------
 * 1. Walk each sentence word-by-word.
 * 2. Skip tokens that contain non-alpha characters (apostrophes,
 *    hyphens, digits) – these are either contractions, compound
 *    words, or numeric values; leave them untouched.
 * 3. For purely alphabetic tokens already present in the
 *    dictionary → correct, keep as-is.
 * 4. Otherwise compute Levenshtein distance against every
 *    dictionary word of similar length; if the closest match
 *    is within MAX_EDIT_DISTANCE (1) → replace.
 * 5. First-character capitalisation of the original is preserved.
 *
 * The dictionary is seeded with an extensive set of common English
 * words plus domain-specific vocabulary (climate / energy / EVs)
 * so that valid words are almost never mis-corrected.
 * ============================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "spelling_correction.h"
#include "preprocessing_types.h"

static char s_dict[DICT_MAX_WORDS][DICT_WORD_LEN];
static int  s_dict_size = 0;

/* ----------------------------------------------------------------
 * Built-in word list
 * Covers: high-frequency English function words, common verbs,
 * adjectives, nouns, and domain vocabulary (climate/energy/EVs).
 * ---------------------------------------------------------------- */
static const char *BUILTIN_WORDS[] = {
    /* --- function words --- */
    "a","an","the","and","or","but","of","in","on","at","to","for",
    "with","by","from","as","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","must","can","its","it",
    "this","that","these","those","their","they","them","we","our",
    "us","you","your","he","she","his","her","him","not","no","nor",
    "so","yet","both","either","neither","each","all","more","most",
    "much","many","some","any","few","other","such","own","same",
    "than","too","very","just","about","over","after","before",
    "during","since","until","while","where","which","who","whom",
    "what","when","how","if","though","although","because","unless",
    "whether","into","onto","upon","against","among","between",
    "through","throughout","without","within","along","across",
    "behind","beyond","below","above","under","around","off","up",
    "out","down","then","there","here","now","never","always",
    "often","usually","already","still","also","even","only","just",
    "once","twice","again","away","back","else","ever","every",
    "another","several","enough","indeed","however","therefore",
    "moreover","furthermore","nevertheless","otherwise","meanwhile",

    /* --- common verbs (base + inflected forms) --- */
    "change","changes","changed","changing",
    "cause","causes","caused","causing",
    "increase","increases","increased","increasing",
    "decrease","decreases","decreased","decreasing",
    "reduce","reduces","reduced","reducing",
    "generate","generates","generated","generating",
    "provide","provides","provided","providing",
    "use","uses","used","using",
    "convert","converts","converted","converting",
    "harness","harnessed","harnessing",
    "affect","affects","affected","affecting",
    "limit","limits","limited","limiting",
    "record","records","recorded","recording",
    "offer","offers","offered","offering",
    "enable","enables","enabled","enabling",
    "create","creates","created","creating",
    "represent","represents","represented",
    "accelerate","accelerates","accelerated",
    "expand","expands","expanded","expanding",
    "install","installs","installed","installing",
    "announce","announces","announced","announcing",
    "integrate","integrates","integrated","integrating",
    "shift","shifts","shifted","shifting",
    "emit","emits","emitted","emitting",
    "deploy","deploys","deployed","deploying",
    "develop","develops","developed","developing",
    "advance","advances","advanced","advancing",
    "invest","invests","invested","investing",
    "power","powers","powered","powering",
    "drive","drives","driven","driving",
    "rise","rises","risen","rising",
    "fall","falls","fallen","falling",
    "grow","grows","grown","growing",
    "burn","burns","burned","burning",
    "melt","melts","melted","melting",
    "warm","warms","warmed","warming",
    "cool","cools","cooled","cooling",
    "store","stores","stored","storing",
    "charge","charges","charged","charging",
    "travel","travels","traveled","travelling",
    "threaten","threatens","threatened",
    "displace","displaces","displaced",
    "disrupt","disrupts","disrupted",
    "adopt","adopts","adopted","adopting",
    "tap","taps","tapped","tapping",
    "spin","spins","spinning","spun",
    "draw","draws","drawing","drawn",
    "make","makes","made","making",
    "take","takes","taken","taking",
    "give","gives","given","giving",
    "come","comes","came","coming",
    "go","goes","went","gone","going",
    "see","sees","saw","seen","seeing",
    "know","knows","knew","known","knowing",
    "think","thinks","thought","thinking",
    "become","becomes","became","becoming",
    "include","includes","included","including",
    "require","requires","required","requiring",
    "expect","expects","expected","expecting",
    "allow","allows","allowed","allowing",
    "move","moves","moved","moving",
    "plan","plans","planned","planning",
    "show","shows","showed","shown","showing",
    "put","puts","putting",
    "keep","keeps","kept","keeping",
    "hold","holds","held","holding",
    "turn","turns","turned","turning",
    "set","sets","setting",
    "run","runs","ran","running",
    "help","helps","helped","helping",
    "work","works","worked","working",
    "build","builds","built","building",
    "improve","improves","improved","improving",
    "manage","manages","managed","managing",
    "face","faces","faced","facing",
    "lose","loses","lost","losing",
    "add","adds","added","adding",
    "lead","leads","led","leading",
    "play","plays","played","playing",
    "reach","reaches","reached","reaching",
    "replace","replaces","replaced","replacing",
    "begin","begins","began","begun","beginning",
    "continue","continues","continued","continuing",
    "produce","produces","produced","producing",
    "supply","supplies","supplied","supplying",
    "support","supports","supported","supporting",
    "protect","protects","protected","protecting",
    "prevent","prevents","prevented","preventing",
    "push","pushes","pushed","pushing",
    "pull","pulls","pulled","pulling",
    "pass","passes","passed","passing",
    "cut","cuts","cutting",
    "drop","drops","dropped","dropping",
    "rise","raises","raised","raising",
    "slow","slows","slowed","slowing",
    "speed","speeds","sped","speeding",

    /* --- common nouns --- */
    "time","year","years","decade","decades","century","centuries",
    "day","days","month","months","hour","hours","minute","minutes",
    "world","country","countries","nation","nations","city","cities",
    "area","areas","region","regions","place","places","site",
    "people","person","man","men","woman","women","child","children",
    "life","lives","home","homes","community","communities",
    "government","governments","policy","policies","law","laws",
    "industry","industries","company","companies","business",
    "economy","economics","market","markets","trade","investment",
    "investment","investments","fund","funds","cost","costs","price",
    "rate","rates","level","levels","amount","amounts","number",
    "part","parts","type","types","kind","kinds","form","forms",
    "way","ways","point","points","case","cases","example","examples",
    "problem","problems","issue","issues","question","questions",
    "result","results","effect","effects","impact","impacts",
    "change","changes","process","processes","system","systems",
    "plan","plans","program","programs","project","projects",
    "data","information","research","study","studies","analysis",
    "model","models","report","reports","review","reviews",
    "source","sources","resource","resources","material","materials",
    "power","energy","fuel","fuels","heat","light","water","air",
    "land","sea","ocean","oceans","river","rivers","lake","lakes",
    "earth","planet","atmosphere","surface","ground","soil",
    "food","plant","plants","animal","animals","species","habitat",
    "gas","gases","particle","particles","molecule","molecules",

    /* --- common adjectives --- */
    "new","old","good","bad","big","small","large","little","long",
    "short","high","low","great","important","major","minor","main",
    "key","primary","secondary","global","local","national","natural",
    "social","economic","political","environmental","scientific",
    "significant","critical","essential","necessary","possible",
    "available","current","recent","future","past","present","next",
    "last","early","late","first","second","third","final","initial",
    "total","full","entire","complete","general","specific","common",
    "public","private","open","free","clean","clear","direct","rapid",
    "rapid","slow","fast","quick","strong","weak","heavy","light",
    "hot","cold","warm","cool","wet","dry","hard","soft","safe","secure",
    "stable","consistent","efficient","effective","sustainable",
    "renewable","reliable","resilient","diverse","unique","complex",
    "simple","basic","advanced","modern","traditional","innovative",
    "industrial","commercial","residential","urban","rural","coastal",
    "widespread","unprecedented","transformative","promising",
    "vulnerable","severe","extreme","alarming","dramatic","rapid",
    "continual","continuous","gradual","steady","growing","rising",
    "falling","increasing","decreasing","expanding","improving",
    "shifting","emerging","existing","leading","following","resulting",

    /* --- common adverbs --- */
    "rapidly","significantly","dramatically","substantially","largely",
    "primarily","mainly","generally","particularly","especially",
    "specifically","directly","largely","mostly","nearly","almost",
    "approximately","effectively","efficiently","consistently",
    "continuously","increasingly","gradually","steadily","widely",
    "globally","locally","nationally","internationally","severely",
    "urgently","critically","currently","recently","previously",
    "heavily","highly","deeply","greatly","strongly","significantly",
    "clearly","simply","easily","quickly","slowly","nearly","fully",

    /* --- climate / environment --- */
    "climate","warming","temperature","temperatures","greenhouse",
    "emission","emissions","carbon","dioxide","methane","fossil",
    "deforestation","revolution","glaciers","polar","caps","sea",
    "coastal","extreme","weather","hurricanes","droughts","floods",
    "wildfires","agricultural","food","insecurity","biodiversity",
    "habitat","destruction","species","environment","sustainability",
    "atmosphere","precipitation","rainfall","drought","ecosystem",
    "ecosystems","ecology","ecological","carbon","pollution","toxic",
    "ozone","radiation","radiation","ice","permafrost","tundra",
    "coral","reef","reefs","wetland","wetlands","forest","forests",
    "woodland","grassland","savanna","desert","arctic","antarctic",
    "glacier","glaciers","iceberg","icebergs","snowpack","snowfall",
    "heatwave","heatwaves","wildfire","storm","storms","hurricane",
    "cyclone","typhoon","tornado","flood","flooding","erosion",
    "extinction","endangered","threatened","conservation","diversity",
    "biome","biosphere","geosphere","hydrosphere","lithosphere",
    "sequestration","mitigation","adaptation","resilience","footprint",
    "offset","offsetting","net","zero","neutral","negative","positive",
    "intergovernmental","panel","celsius","fahrenheit","degree",
    "degrees","threshold","target","limit","cap","benchmark","goal",

    /* --- energy --- */
    "energy","renewable","solar","wind","hydro","hydropower","power",
    "geothermal","photovoltaic","turbine","turbines","electricity",
    "grid","grids","smart","battery","batteries","storage","lithium",
    "ion","discharge","capacity","infrastructure","generation",
    "generator","megawatt","gigawatt","kilowatt","watt","panel",
    "panels","array","arrays","farm","farms","station","stations",
    "plant","plants","reactor","reactors","transmission","distribution",
    "load","demand","supply","peak","baseload","dispatchable",
    "intermittent","variable","curtailment","flexibility","stability",
    "microgrid","nanogrid","substation","transformer","inverter",
    "converter","meter","meters","sensor","sensors","monitoring",
    "thermal","kinetic","potential","chemical","electrical","nuclear",
    "coal","natural","petroleum","crude","oil","gas","propane","diesel",
    "biomass","biogas","biofuel","ethanol","methanol","hydrogen",
    "collection","harnessing","tapping","extraction","refining",
    "drilling","pipeline","pipelines","tank","tanks","reservoir",

    /* --- electric vehicles / transport --- */
    "electric","vehicle","vehicles","transport","transportation",
    "motor","motors","rechargeable","pack","packs","range","miles",
    "distance","solid","state","technology","technologies",
    "automaker","automakers","combustion","engine","engines",
    "incentives","credits","subsidies","adoption","mobility",
    "synergistic","synergy","integration","roadmap","phase","out",
    "highway","road","roads","street","streets","lane","lanes",
    "traffic","commute","commuting","fleet","fleets","bus","buses",
    "truck","trucks","car","cars","van","vans","motorcycle","bicycle",
    "rail","train","trains","subway","tram","ferry","aircraft","ship",
    "charging","station","network","connector","plug","socket","port",
    "kilowatt","hour","kilowatt-hour","discharge","cycle","cycles",
    "capacity","degradation","lifespan","warranty","manufacturer",
    "model","sedan","suv","pickup","hatchback","crossover","coupe",
    "autonomous","self-driving","connected","shared","mobility",
    "infrastructure","deployment","rollout","penetration","market",
    "share","adoption","uptake","consumer","consumers","user","users",
    "driver","drivers","passenger","passengers","commuter","commuters",
    "practical","everyday","public","private","rapid","slow",
    "overnight","fast-charging","supercharger","destination",


    /* --- commonly missed words (false-positive guard) --- */
    "warns","warn","warned","warning",
    "flowing","flows","flow","flowed",
    "loss","losses","loses","alone",
    "harnesses","harnessing","harnessed",
    "times","rates","cases","forms","farms",
    "climatic","unprecedented","predominantly",
    "viable","viable","synergistic","mobility",
    "section","sector","capture","captured",
    "transition","transitions","transitioned",
    "urgently","urgency","mitigate","mitigation",
    "replenish","replenishes","replenished",
    "displacement","disruption","disruptions",
    "modernization","modernisation","diversification",
    "electrification","decarbonization","decarbonisation",
    "photovoltaic","geothermal","hydroelectric",
    "interconnected","interdependent","transformative",
    "consistently","continuously","significantly",
    "predominantly","approximately","internationally",

    NULL  /* sentinel */
};

/* ---- helpers ---- */
static void to_lower(const char *src, char *dst, int max) {
    int i = 0;
    while (src[i] && i < max - 1) { dst[i] = (char)tolower((unsigned char)src[i]); i++; }
    dst[i] = '\0';
}

static int is_pure_alpha(const char *s) {
    for (; *s; s++) if (!isalpha((unsigned char)*s)) return 0;
    return 1;
}

static void dict_add(const char *word) {
    if (s_dict_size >= DICT_MAX_WORDS) return;
    char lw[DICT_WORD_LEN];
    to_lower(word, lw, DICT_WORD_LEN);
    for (int i = 0; i < s_dict_size; i++)
        if (strcmp(s_dict[i], lw) == 0) return;
    strncpy(s_dict[s_dict_size], lw, DICT_WORD_LEN - 1);
    s_dict[s_dict_size][DICT_WORD_LEN - 1] = '\0';
    s_dict_size++;
}

static int dict_contains(const char *lw) {
    for (int i = 0; i < s_dict_size; i++)
        if (strcmp(s_dict[i], lw) == 0) return 1;
    return 0;
}

/* Levenshtein distance (two-row iterative) */
static int levenshtein(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b);
    if (la == 0) return lb;
    if (lb == 0) return la;
    if (la >= DICT_WORD_LEN || lb >= DICT_WORD_LEN) return 9999;

    static int row0[DICT_WORD_LEN + 1], row1[DICT_WORD_LEN + 1];
    for (int j = 0; j <= lb; j++) row0[j] = j;
    for (int i = 1; i <= la; i++) {
        row1[0] = i;
        for (int j = 1; j <= lb; j++) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            int d = row0[j] + 1, ins = row1[j-1] + 1, sub = row0[j-1] + cost;
            row1[j] = d < ins ? (d < sub ? d : sub) : (ins < sub ? ins : sub);
        }
        for (int j = 0; j <= lb; j++) row0[j] = row1[j];
    }
    return row0[lb];
}

/* ================================================================
 * sc_load_dictionary()
 * ================================================================ */
int sc_load_dictionary(const char *path) {
    if (s_dict_size == 0)
        for (int i = 0; BUILTIN_WORDS[i]; i++) dict_add(BUILTIN_WORDS[i]);

    if (!path) return s_dict_size;
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[spell] Cannot open dict: %s\n", path); return -1; }
    char line[DICT_WORD_LEN];
    int added = 0;
    while (fgets(line, sizeof(line), f)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len]='\0';
        if (len > 0) { dict_add(line); added++; }
    }
    fclose(f);
    return added;
}

/* ================================================================
 * sc_correct_word()
 * ================================================================ */
const char *sc_correct_word(const char *word) {
    if (!word || !word[0]) return word;
    if (s_dict_size == 0) sc_load_dictionary(NULL);

    /* Only correct purely alphabetic tokens */
    if (!is_pure_alpha(word)) return word;

    /* Very short words (≤ 3 chars) are almost always correct */
    if (strlen(word) <= 3) return word;

    char lower[DICT_WORD_LEN];
    to_lower(word, lower, DICT_WORD_LEN);

    if (dict_contains(lower)) return word;  /* already correct */

    int         best_dist = MAX_EDIT_DISTANCE + 1;
    int         best_idx  = -1;
    int         wlen      = (int)strlen(lower);

    for (int i = 0; i < s_dict_size; i++) {
        int dlen = (int)strlen(s_dict[i]);
        /* Length filter: only compare words within MAX_EDIT_DISTANCE length */
        if (abs(dlen - wlen) > MAX_EDIT_DISTANCE) continue;
        int d = levenshtein(lower, s_dict[i]);
        if (d < best_dist) { best_dist = d; best_idx = i; }
        if (best_dist == 0) break;
    }

    if (best_idx >= 0 && best_dist <= MAX_EDIT_DISTANCE)
        return s_dict[best_idx];

    return word;  /* no close match — leave unchanged */
}

/* ================================================================
 * correct_spelling_in_list()
 * ================================================================ */
int correct_spelling_in_list(SentenceList *sl) {
    if (!sl) return PP_ERR_NULL;
    if (s_dict_size == 0) sc_load_dictionary(NULL);

    char corrected[MAX_SENTENCE_LEN];
    char token[MAX_WORD_LEN];

    for (int s = 0; s < sl->count; s++) {
        char *src    = sl->data[s];
        char *dst    = corrected;
        char *dstend = corrected + MAX_SENTENCE_LEN - 1;

        while (*src) {
            while (*src && isspace((unsigned char)*src)) {
                if (dst < dstend) *dst++ = *src;
                src++;
            }
            if (!*src) break;

            int tlen = 0;
            while (*src && !isspace((unsigned char)*src)) {
                if (tlen < MAX_WORD_LEN - 1) token[tlen++] = *src;
                src++;
            }
            token[tlen] = '\0';

            const char *out = sc_correct_word(token);

            /* Preserve original leading capitalisation */
            char adjusted[DICT_WORD_LEN];
            if (isupper((unsigned char)token[0]) && islower((unsigned char)out[0])) {
                strncpy(adjusted, out, DICT_WORD_LEN - 1);
                adjusted[0] = (char)toupper((unsigned char)adjusted[0]);
                out = adjusted;
            }

            size_t olen = strlen(out);
            if (dst + olen < dstend) { memcpy(dst, out, olen); dst += olen; }
        }
        *dst = '\0';
        strncpy(sl->data[s], corrected, MAX_SENTENCE_LEN - 1);
        sl->data[s][MAX_SENTENCE_LEN - 1] = '\0';
    }
    return PP_OK;
}

/* ================================================================
 * sc_print_dictionary()
 * ================================================================ */
void sc_print_dictionary(void) {
    printf("Dictionary: %d word(s)\n", s_dict_size);
    for (int i = 0; i < s_dict_size; i++)
        printf("  [%04d] %s\n", i, s_dict[i]);
}
