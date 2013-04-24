#include <getopt.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "evalue.h"
#include "mpi_module.h"
#include "swsharp/swsharp.h"

static struct option options[] = {
    {"cards", required_argument, 0, 'c'},
    {"gap-extend", required_argument, 0, 'e'},
    {"gap-open", required_argument, 0, 'g'},
    {"query", required_argument, 0, 'i'},
    {"target", required_argument, 0, 'j'},
    {"matrix", required_argument, 0, 'm'},
    {"out", required_argument, 0, 'o'},
    {"evalue", required_argument, 0, 'E'},
    {"max-aligns", required_argument, 0, 'M'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static void help();

static void valueFunction(float* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param);

static int chainCmp(const void* a_, const void* b_);

int main(int argc, char* argv[]) {

    int rank;
    int nodes;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    
    char* queryPath = NULL;
    char* databasePath = NULL;

    int gapOpen = 10;
    int gapExtend = 2;
    
    char* matrix = BLOSUM_62;
        
    int maxAlignments = 10;
    float maxEValue = 1000;
    
    int cardsLen = -1;
    int* cards = NULL;
    
    int i;
    char* out = NULL;
    
    while (1) {

        char argument = getopt_long(argc, argv, "i:j:g:e:", options, NULL);

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
            cardsLen = strlen(optarg);
            for (i = 0; i < cardsLen; ++i) cards[i] = optarg[i] - '0';
            break;
        case 'o':
            out = optarg;
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
    
    ASSERT_CALL(queryPath != NULL, help, "missing option -i (query file)");
    ASSERT_CALL(databasePath != NULL, help, "missing option -j (database file)");
    
    if (cardsLen == -1) {
        cudaGetCards(&cards, &cardsLen);
    }
    
    ASSERT(cudaCheckCards(cards, cardsLen), "invalid cuda cards");
    
    ASSERT_CALL(gapExtend > 0 && gapExtend <= gapOpen, help, "invalid gap extend");
    ASSERT_CALL(maxEValue > 0, help, "invalid evalue");
    
    Scorer* scorer;
    scorerCreateMatrix(&scorer, matrix, gapOpen, gapExtend);
    
    Chain** queries = NULL;
    int queriesLen = 0;
    readFastaChains(&queries, &queriesLen, queryPath);
    
    Chain** database = NULL; 
    int databaseLen = 0;
    readFastaChains(&database, &databaseLen, databasePath);
    
    // MPI sort db for performance
    qsort(database, databaseLen, sizeof(Chain*), chainCmp);
    
    ChainDatabase* chainDatabase = chainDatabaseCreate(database, databaseLen);
    
    // MPI create dummy indexes
    int* indexes = (int*) malloc(databaseLen * sizeof(int));
    for (i = 0; i < databaseLen; ++i) {
        indexes[i] = i;
    }
    
    // MPI calculate indexes to solve
    int indexesOffset = rank * (databaseLen / nodes);
    int lastNode = rank == nodes - 1;
    int indexesLen = lastNode ? databaseLen - indexesOffset : databaseLen / nodes;

    DbAlignment*** dbAlignments;
    int* dbAlignmentsLen;

    shotgunDatabase(&dbAlignments, &dbAlignmentsLen, SW_ALIGN, queries, 
        queriesLen, chainDatabase, scorer, maxAlignments, valueFunction, 
        (void*) scorer, maxEValue, indexes + indexesOffset, indexesLen, cards, 
        cardsLen, NULL);
        
    // master node gathers and outputs data
    if (rank == MASTER_NODE) {
    
        // recieve and join
        gatherMpiData(&dbAlignments, &dbAlignmentsLen, queries, queriesLen, 
            database, databaseLen, scorer, maxAlignments);

        // output
        outputShotgunDatabase(dbAlignments, dbAlignmentsLen, queriesLen, 
            out, SW_OUT_DB_BLASTM9);
    } else {
        // send data to master node
        sendMpiData(dbAlignments, dbAlignmentsLen, queries, queriesLen, 
            database, databaseLen);
    }
    
    free(indexes);
    
    deleteShotgunDatabase(dbAlignments, dbAlignmentsLen, queriesLen);
    
    chainDatabaseDelete(chainDatabase);
    
    deleteFastaChains(queries, queriesLen);
    deleteFastaChains(database, databaseLen);
    
    scorerDelete(scorer);
    
    free(cards);

    MPI_Finalize();

    return 0;
}

static void valueFunction(float* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param) {
    eValues(values, scores, query, database, databaseLen, (Scorer*) param);
}

static int chainCmp(const void* a_, const void* b_) {

    Chain* a = *((Chain**) a_);
    Chain* b = *((Chain**) b_);

    return chainGetLength(a) - chainGetLength(b);
}

static void help() {
    printf("help\n");
}
