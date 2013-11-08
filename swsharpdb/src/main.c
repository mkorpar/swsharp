#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "swsharp/evalue.h"
#include "swsharp/swsharp.h"

#define ASSERT(expr, fmt, ...)\
    do {\
        if (!(expr)) {\
            fprintf(stderr, "[ERROR]: " fmt "\n", ##__VA_ARGS__);\
            exit(-1);\
        }\
    } while(0)

#define CHAR_INT_LEN(x) (sizeof(x) / sizeof(CharInt))

typedef struct CharInt {
    const char* format;
    const int code;
} CharInt;

typedef struct ValueFunctionParam {
    Scorer* scorer;
    int totalLength;
} ValueFunctionParam;

static struct option options[] = {
    {"cards", required_argument, 0, 'c'},
    {"gap-extend", required_argument, 0, 'e'},
    {"gap-open", required_argument, 0, 'g'},
    {"query", required_argument, 0, 'i'},
    {"target", required_argument, 0, 'j'},
    {"matrix", required_argument, 0, 'm'},
    {"out", required_argument, 0, 'o'},
    {"outfmt", required_argument, 0, 't'},
    {"evalue", required_argument, 0, 'E'},
    {"max-aligns", required_argument, 0, 'M'},
    {"algorithm", required_argument, 0, 'A'},
    {"cache", no_argument, 0, 'C'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static CharInt outFormats[] = {
    { "bm0", SW_OUT_DB_BLASTM0 },
    { "bm8", SW_OUT_DB_BLASTM8 },
    { "bm9", SW_OUT_DB_BLASTM9 },
    { "light", SW_OUT_DB_LIGHT }
};

static CharInt algorithms[] = {
    { "SW", SW_ALIGN },
    { "NW", NW_ALIGN },
    { "HW", HW_ALIGN },
    { "OV", OV_ALIGN }
};

static void help();

static void getCudaCards(int** cards, int* cardsLen, char* optarg);

static int getOutFormat(char* optarg);
static int getAlgorithm(char* optarg);

static void valueFunction(double* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param);

int main(int argc, char* argv[]) {

    char* queryPath = NULL;
    char* databasePath = NULL;

    int gapOpen = 10;
    int gapExtend = 1;
    
    char* matrix = "BLOSUM_62";
        
    int maxAlignments = 10;
    float maxEValue = 10;
    
    int cardsLen = -1;
    int* cards = NULL;
    
    char* out = NULL;
    int outFormat = SW_OUT_DB_BLASTM9;

    int algorithm = SW_ALIGN;
    
    int cache = 0;

    while (1) {

        char argument = getopt_long(argc, argv, "i:j:g:e:h", options, NULL);

        if (argument == -1) {
            break;
        }

        switch (argument) {
        case 'i':
            queryPath = optarg;
            break;
        case 'j':
            databasePath = optarg;
            break;
        case 'g':
            gapOpen = atoi(optarg);
            break;
        case 'e':
            gapExtend = atoi(optarg);
            break;
        case 'c':
            getCudaCards(&cards, &cardsLen, optarg);
            break;
        case 'o':
            out = optarg;
            break;
        case 't':
            outFormat = getOutFormat(optarg);
            break;
        case 'M':
            maxAlignments = atoi(optarg);
            break;
        case 'E':
            maxEValue = atof(optarg);
            break;
        case 'm':
            matrix = optarg;
            break;
        case 'A':
            algorithm = getAlgorithm(optarg);
            break;
        case 'C':
            cache = 1;
            break;
        case 'h':
        default:
            help();
            return -1;
        }
    }
    
    ASSERT(queryPath != NULL, "missing option -i (query file)");
    ASSERT(databasePath != NULL, "missing option -j (database file)");
    
    if (cardsLen == -1) {
        cudaGetCards(&cards, &cardsLen);
    }
    
    ASSERT(cudaCheckCards(cards, cardsLen), "invalid cuda cards");
    
    ASSERT(maxEValue > 0, "invalid evalue");
    
    Scorer* scorer;
    scorerCreateMatrix(&scorer, matrix, gapOpen, gapExtend);
    
    Chain** queries = NULL;
    int queriesLen = 0;
    readFastaChains(&queries, &queriesLen, queryPath);
    
    Chain** database = NULL; 
    int databaseLen = 0;
    readFastaChains(&database, &databaseLen, databasePath);
    
    if (cache) {
        dumpFastaChains(database, databaseLen, databasePath);
    }

    threadPoolInitialize(cardsLen + 8);

    EValueParams* eValueParams = createEValueParams(database, databaseLen, scorer);

    DbAlignment*** dbAlignments = NULL;
    int* dbAlignmentsLens = NULL;

    int databaseCur = 0;
    int databaseStart = 0;
    
    int databaseIdx;
    for (databaseIdx = 0; databaseIdx < databaseLen; ++databaseIdx) {
    
        databaseCur += chainGetLength(database[databaseIdx]);

        if (databaseCur < 400 * 1024 * 1024 && databaseIdx != databaseLen - 1) {
            continue;
        }
        
        ChainDatabase* chainDatabase = chainDatabaseCreate(database, 
            databaseStart, databaseIdx - databaseStart + 1, cards, cardsLen);

        DbAlignment*** dbAlignmentsPart = NULL;
        int* dbAlignmentsPartLens = NULL;

        shotgunDatabase(&dbAlignmentsPart, &dbAlignmentsPartLens, algorithm, 
            queries, queriesLen, chainDatabase, scorer, maxAlignments, valueFunction, 
            (void*) eValueParams, maxEValue, NULL, 0, cards, cardsLen, NULL);
        
        if (dbAlignments == NULL) {
            dbAlignments = dbAlignmentsPart;
            dbAlignmentsLens = dbAlignmentsPartLens;
        } else {
            dbAlignmentsMerge(dbAlignments, dbAlignmentsLens, dbAlignmentsPart, 
                dbAlignmentsPartLens, queriesLen, maxAlignments);
            deleteShotgunDatabase(dbAlignmentsPart, dbAlignmentsPartLens, queriesLen);
        }

        chainDatabaseDelete(chainDatabase);
            
        databaseStart = databaseIdx + 1;
        databaseCur = 0;
    }
    
    outputShotgunDatabase(dbAlignments, dbAlignmentsLens, queriesLen, out, outFormat);
    deleteShotgunDatabase(dbAlignments, dbAlignmentsLens, queriesLen);
    
    deleteEValueParams(eValueParams);
    
    deleteFastaChains(queries, queriesLen);
    deleteFastaChains(database, databaseLen);
    
    scorerDelete(scorer);

    threadPoolTerminate();
    free(cards);
    
    return 0;
}

static void getCudaCards(int** cards, int* cardsLen, char* optarg) {

    *cardsLen = strlen(optarg);
    *cards = (int*) malloc(*cardsLen * sizeof(int));
    
    int i;
    for (i = 0; i < *cardsLen; ++i) {
        (*cards)[i] = optarg[i] - '0';
    }
}

static int getOutFormat(char* optarg) {

    int i;
    for (i = 0; i < CHAR_INT_LEN(outFormats); ++i) {
        if (strcmp(outFormats[i].format, optarg) == 0) {
            return outFormats[i].code;
        }
    }

    ASSERT(0, "unknown out format %s", optarg);
}

static int getAlgorithm(char* optarg) {

    int i;
    for (i = 0; i < CHAR_INT_LEN(algorithms); ++i) {
        if (strcmp(algorithms[i].format, optarg) == 0) {
            return algorithms[i].code;
        }
    }

    ASSERT(0, "unknown algorithm %s", optarg);
}

static void valueFunction(double* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param_ ) {
    
    EValueParams* eValueParams = (EValueParams*) param_;
    eValues(values, scores, query, database, databaseLen, eValueParams);
}

static void help() {
    printf(
    "usage: swsharpdb -i <query db file> -j <target db file> [arguments ...]\n"
    "\n"
    "arguments:\n"
    "    -i, --query <file>\n"
    "        (required)\n"
    "        input fasta database query file\n"
    "    -j, --target <file>\n"
    "        (required)\n"
    "        input fasta database target file\n"
    "    -g, --gap-open <int>\n"
    "        default: 10\n"
    "        gap opening penalty, must be given as a positive integer \n"
    "    -e, --gap-extend <int>\n"
    "        default: 1\n"
    "        gap extension penalty, must be given as a positive integer and\n"
    "        must be less or equal to gap opening penalty\n" 
    "    --matrix <string>\n"
    "        default: BLOSUM_62\n"
    "        similarity matrix, can be one of the following:\n"
    "            BLOSUM_45\n"
    "            BLOSUM_50\n"
    "            BLOSUM_62\n"
    "            BLOSUM_80\n"
    "            BLOSUM_90\n"
    "            BLOSUM_30\n"
    "            BLOSUM_70\n"
    "            BLOSUM_250\n"
    "            EDNA_FULL\n"
    "    --evalue <float>\n"
    "        default: 10.0\n"
    "        evalue threshold, alignments with higher evalue are filtered,\n"
    "        must be given as a positive float\n"
    "    --max-aligns <int>\n"
    "        default: 10\n"
    "        maximum number of alignments to be outputted\n"
    "    --algorithm <string>\n"
    "        default: SW\n"
    "        algorithm used for alignment, must be one of the following: \n"
    "            SW - Smith-Waterman local alignment\n"
    "            NW - Needleman-Wunsch global alignment\n"
    "            HW - semiglobal alignment\n"
    "            OV - overlap alignment\n"
    "    --cards <ints>\n"
    "        default: all available CUDA cards\n"
    "        list of cards should be given as an array of card indexes delimited with\n"
    "        nothing, for example usage of first two cards is given as --cards 01\n"
    "    --out <string>\n"
    "        default: stdout\n"
    "        output file for the alignment\n"
    "    --outfmt <string>\n"
    "        default: bm9\n"
    "        out format for the output file, must be one of the following:\n"
    "            bm0      - blast m0 output format\n"
    "            bm8      - blast m8 tabular output format\n"
    "            bm9      - blast m9 commented tabular output format\n"
    "            light    - score-name tabbed output\n"
    "    --cache\n"
    "        serialized database is stored to speed up future runs with the\n"
    "        same database\n"
    "    -h, -help\n"
    "        prints out the help\n");
}
