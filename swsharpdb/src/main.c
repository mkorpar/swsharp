#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "evalue.h"
#include "swsharp/swsharp.h"

#define OUT_FORMATS_LEN (sizeof(outFormats) / sizeof(CharInt))

typedef struct CharInt {
    const char* format;
    const int code;
} CharInt;

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
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static CharInt outFormats[] = {
    { "bm1", SW_OUT_DB_BLASTM1 },
    { "bm8", SW_OUT_DB_BLASTM8 },
    { "bm9", SW_OUT_DB_BLASTM9 },
    { "light", SW_OUT_DB_LIGHT }
};

static void help();

static void getCudaCards(int** cards, int* cardsLen, char* optarg);

static int getOutFormat(char* optarg);

static void valueFunction(float* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param);

int main(int argc, char* argv[]) {

    char* queryPath = NULL;
    char* databasePath = NULL;

    int gapOpen = 10;
    int gapExtend = 2;
    
    char* matrix = BLOSUM_62;
        
    int maxAlignments = 10;
    float maxEValue = 10;
    
    int cardsLen = -1;
    int* cards = NULL;
    
    char* out = NULL;
    int outFormat = SW_OUT_DB_BLASTM9;

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
    
    ASSERT(gapExtend > 0 && gapExtend <= gapOpen, "invalid gap extend");
    ASSERT(maxEValue > 0, "invalid evalue");
    
    Scorer* scorer;
    scorerCreateMatrix(&scorer, matrix, gapOpen, gapExtend);
    
    Chain** queries = NULL;
    int queriesLen = 0;
    readFastaChains(&queries, &queriesLen, queryPath);
    
    Chain** database = NULL; 
    int databaseLen = 0;
    readFastaChains(&database, &databaseLen, databasePath);
    
    ChainDatabase* chainDatabase = chainDatabaseCreate(database, databaseLen);

    DbAlignment*** dbAlignments;
    int* dbAlignmentsLen;

    shotgunDatabase(&dbAlignments, &dbAlignmentsLen, SW_ALIGN, queries, 
        queriesLen, chainDatabase, scorer, maxAlignments, valueFunction, 
        (void*) scorer, maxEValue, NULL, 0, cards, cardsLen, NULL);
      
    outputShotgunDatabase(dbAlignments, dbAlignmentsLen, queriesLen, 
        out, outFormat);
    
    deleteShotgunDatabase(dbAlignments, dbAlignmentsLen, queriesLen);

    chainDatabaseDelete(chainDatabase);
    
    deleteFastaChains(queries, queriesLen);
    deleteFastaChains(database, databaseLen);
    
    scorerDelete(scorer);
    
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
    for (i = 0; i < OUT_FORMATS_LEN; ++i) {
        if (strcmp(outFormats[i].format, optarg) == 0) {
            return outFormats[i].code;
        }
    }

    ERROR("unknown out format %s", optarg);
}

static void valueFunction(float* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param) {
    eValues(values, scores, query, database, databaseLen, (Scorer*) param);
}

static void help() {
    printf("help\n");
}
