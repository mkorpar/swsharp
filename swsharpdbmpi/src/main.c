#include <getopt.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "database_utils.h"
#include "evalue.h"
#include "mpi_module.h"
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
    { "HW", HW_ALIGN }
};

static void help();

static void getCudaCards(int** cards, int* cardsLen, char* optarg);

static int getOutFormat(char* optarg);
static int getAlgorithm(char* optarg);

static void valueFunction(double* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param);

int main(int argc, char* argv[]) {

    int mpiRank = 0;
    int mpiNodes = 1;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiNodes);

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
    
    ASSERT(gapOpen > 0, "invalid gap open");
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
    
    EValueParams* eValueParams = createEValueParams(database, databaseLen, 
        scorer);

    // mpi data
    int mpiDbStep = databaseLen / mpiNodes;
    int mpiDbOff = mpiRank * mpiDbStep;
    int mpiDbLen = mpiRank == mpiNodes - 1 ? databaseLen - mpiDbOff : mpiDbStep;

    ChainDatabase* chainDatabase = chainDatabaseCreate(database, mpiDbOff, mpiDbLen);

    DbAlignment*** dbAlignments;
    int* dbAlignmentsLens;

    shotgunDatabase(&dbAlignments, &dbAlignmentsLens, algorithm, queries, 
        queriesLen, chainDatabase, scorer, maxAlignments, valueFunction, 
        (void*) eValueParams, maxEValue, NULL, 0, cards, cardsLen, NULL);

    // master node gathers and outputs data
    int masterNode = 0;
    if (mpiRank == masterNode) {

        int i;
        size_t size;
        
        size = mpiNodes * sizeof(DbAlignment***);
        DbAlignment**** dbAlignmentsMpi = (DbAlignment****) malloc(size); 
        
        size = mpiNodes * sizeof(int**);
        int** dbAlignmentsLensMpi = (int**) malloc(size);
         
        size = mpiNodes * sizeof(int*);
        int* dbAlignmentsLenMpi = (int*) malloc(size);
        
        dbAlignmentsMpi[mpiRank] = dbAlignments;
        dbAlignmentsLensMpi[mpiRank] = dbAlignmentsLens;
        dbAlignmentsLenMpi[mpiRank] = queriesLen;
        
        for (i = 0; i < mpiNodes; ++i) {
            if (i != mpiRank) {
                recieveMpiData(&(dbAlignmentsMpi[i]), &(dbAlignmentsLensMpi[i]), 
                    &(dbAlignmentsLenMpi[i]), queries, database, scorer, i);
            }
        }
        
        int dbAlignmentsLen;
        joinShotgunDatabases(&dbAlignments, &dbAlignmentsLens, &dbAlignmentsLen, 
            dbAlignmentsMpi, dbAlignmentsLensMpi, dbAlignmentsLenMpi, 
            mpiNodes, maxAlignments);
           
        // output
        outputShotgunDatabase(dbAlignments, dbAlignmentsLens, dbAlignmentsLen, 
            out, outFormat);
        
        // delete real data
        for (i = 0; i < mpiNodes; ++i) {
            deleteShotgunDatabase(dbAlignmentsMpi[i], dbAlignmentsLensMpi[i], 
                dbAlignmentsLenMpi[i]);
        }
        
        // delete placeholders
        for (i = 0; i < dbAlignmentsLen; ++i) {
            free(dbAlignments[i]);
        }
        free(dbAlignments);
        free(dbAlignmentsLens);
        
        free(dbAlignmentsMpi);
        free(dbAlignmentsLensMpi);
        free(dbAlignmentsLenMpi);
        
    } else {
        sendMpiData(dbAlignments, dbAlignmentsLens, queriesLen, masterNode);
        deleteShotgunDatabase(dbAlignments, dbAlignmentsLens, queriesLen);
    }
        
    chainDatabaseDelete(chainDatabase);

    deleteEValueParams(eValueParams);
    
    deleteFastaChains(queries, queriesLen);
    deleteFastaChains(database, databaseLen);
    
    scorerDelete(scorer);
    
    free(cards);
    
    MPI_Finalize();
    
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
    "    -h, -help\n"
    "        prints out the help\n");
}
