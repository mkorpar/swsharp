/*
swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and 
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Šikić

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Contact the author by mkorpar@gmail.com.
*/

#ifdef __CUDACC__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "cpu_module.h"
#include "cuda_utils.h"
#include "error.h"
#include "scorer.h"
#include "thread.h"
#include "threadpool.h"
#include "utils.h"

#include "score_database_gpu_short.h"

#define MAX_CPU_LEN         2000
#define CPU_WORKERS         8
#define CPU_WORKER_STEP     32

#define THREADS   128
#define BLOCKS    120

#define INT4_ZERO make_int4(0, 0, 0, 0)
#define INT4_SCORE_MIN make_int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

typedef struct GpuDatabase {
    int card;
    int* offsets;
    int* lengths;
    int* lengthsPadded;
    cudaArray* sequences;
    int* indexes;
    int* scores;
    int2* hBus;
} GpuDatabase;

typedef struct GpuDatabaseContext {
    int card;
    int length;
    int blocks;
    int* offsets;
    size_t offsetsSize;
    int* lengths;
    int* lengthsPadded;
    size_t lengthsSize;
    char4* sequences;
    int sequencesCols;
    int sequencesRows;
    size_t sequencesSize;
    int* indexes;
    size_t indexesSize;
    GpuDatabase* gpuDatabase;
} GpuDatabaseContext;

struct ShortDatabase {
    Chain** database;
    int databaseLen;
    int length;
    int* positions;
    int* order;
    int* indexes;
    int blocks;
    int sequencesRows;
    int sequencesCols;
    GpuDatabase* gpuDatabases;
    int gpuDatabasesLen;
};

typedef struct Context {
    int* scores; 
    int type;
    Chain** queries;
    int queriesLen;
    ShortDatabase* shortDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int* cards;
    int cardsLen;
} Context;

typedef struct QueryProfile {
    int height;
    int width;
    int length;
    char4* data;
    size_t size;
} QueryProfile;

typedef struct QueryProfileGpu {
    cudaArray* data;
} QueryProfileGpu;

typedef void (*ScoringFunction)(int*, int2*, int*, int*, int*, int*, int);

typedef struct KernelContext {
    int* scores;
    int type;
    ScoringFunction scoringFunction;
    QueryProfile* queryProfile;
    Chain* query;
    ShortDatabase* shortDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int card;
} KernelContext;

typedef struct KernelContexts {
    KernelContext* contexts;
    int contextsLen;
    long long cells;
} KernelContexts;

typedef struct KernelContextCpu {
    int* scores;
    int type;
    Chain* query;
    ShortDatabase* shortDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int* lastIndexSolvedCpu;
    int* firstIndexSolvedGpu;
    Mutex* indexSolvedMutex;
} KernelContextCpu;

typedef struct CpuWorkerContext {
    int* scores;
    int type;
    Chain* query;
    Chain** database;
    int databaseLen;
    Scorer* scorer;
    int* lastIndexSolvedCpu;
    int* firstIndexSolvedGpu;
    Mutex* indexSolvedMutex;
} CpuWorkerContext;

static __constant__ int gapOpen_;
static __constant__ int gapExtend_;

static __constant__ int rows_;
static __constant__ int rowsPadded_;
static __constant__ int width_;
static __constant__ int length_;

texture<int, 2, cudaReadModeElementType> seqsTexture;
texture<char4, 2, cudaReadModeElementType> qpTexture;

//******************************************************************************
// PUBLIC

extern ShortDatabase* shortDatabaseCreate(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen);

extern void shortDatabaseDelete(ShortDatabase* shortDatabase);

extern void scoreShortDatabaseGpu(int* scores, int type, Chain* query, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread);

extern void scoreShortDatabasesGpu(int* scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// constructor
static ShortDatabase* createDatabase(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen);

// gpu constructor thread
static void* createDatabaseGpu(void* param);

// destructor
static void deleteDatabase(ShortDatabase* database);

// scoring 
static void scoreDatabase(int* scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

static void* scoreDatabaseThread(void* param);

static void scoreDatabaseMulti(int* scores, int type,
    ScoringFunction scoringFunction, Chain** queries, int queriesLen, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen);

static void scoreDatabaseSingle(int* scores, int type,
    ScoringFunction scoringFunction, Chain** queries, int queriesLen, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen);

// cpu kernels 
static void* kernelThread(void* param);

static void* kernelsThread(void* param);

static void* kernelThreadCpu(void* param);

static void* cpuWorker(void* param);

// gpu kernels 
__global__ static void hwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block);
    
__global__ static void nwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block);

__global__ static void ovSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block);

__global__ static void swSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block);
    
// query profile
static QueryProfile* createQueryProfile(Chain* query, Scorer* scorer);

static void deleteQueryProfile(QueryProfile* queryProfile);

static QueryProfileGpu* createQueryProfileGpu(QueryProfile* queryProfile);

static void deleteQueryProfileGpu(QueryProfileGpu* queryProfileGpu);

// utils
static int int2CmpY(const void* a_, const void* b_);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern ShortDatabase* shortDatabaseCreate(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen) {
    return createDatabase(database, databaseLen, minLen, maxLen, cards, cardsLen);
}
    
extern void shortDatabaseDelete(ShortDatabase* shortDatabase) {
    deleteDatabase(shortDatabase);
}

extern size_t shortDatabaseGpuMemoryConsumption(Chain** database,
    int databaseLen, int minLen, int maxLen) {

    int length = 0;
    int maxHeight = 0;

    for (int i = 0; i < databaseLen; ++i) {

        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            length++;
            maxHeight = max(maxHeight, n);
        }
    }

    if (length == 0) {
        return 0;
    }

    maxHeight = (maxHeight >> 2) + ((maxHeight & 3) > 0);

    int sequencesCols = THREADS * BLOCKS;

    int blocks = length / sequencesCols + (length % sequencesCols > 0);
    int hBusHeight = maxHeight * 4;

    //##########################################################################

    const int bucketDiff = 32;
    int bucketsLen = maxLen / bucketDiff + (maxLen % bucketDiff > 0);

    int* buckets = (int*) malloc(bucketsLen * sizeof(int));
    memset(buckets, 0, bucketsLen * sizeof(int));

    for (int i = 0; i < databaseLen; ++i) {

        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            buckets[n >> 5]++;
        }
    }

    int sequencesRows = 0;
    for (int i = 0, j = 0; i < bucketsLen; ++i) {
        
        j += buckets[i];

        int d = j / sequencesCols;
        int r = j % sequencesCols;

        sequencesRows += d * ((i + 1) * (bucketDiff / 4));
        j = r;

        if (i == bucketsLen - 1 && j > 0) {
            sequencesRows += ((i + 1) * (bucketDiff / 4));
        }
    }

    free(buckets);

    /*
    int* lengths = (int*) malloc(length * sizeof(int));

    for (int i = 0, j = 0; i < databaseLen; ++i) {

        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            lengths[j++] = n;
        }
    }

    qsort(lengths, length, sizeof(int), intCmp);
    

    int sequencesRows = 0;

    for (int i = sequencesCols - 1; i < length; i += sequencesCols) {
        int n = lengths[i];
        sequencesRows += (n >> 2) + ((n & 3) > 0);
    }

    if (length % sequencesCols != 0) {
        sequencesRows += maxHeight;
    }

  
  free(lengths);
    */

    //##########################################################################

    size_t hBusSize = sequencesCols * hBusHeight * sizeof(int2);
    size_t offsetsSize = blocks * sizeof(int);
    size_t lengthsSize = blocks * sequencesCols * sizeof(int);
    size_t sequencesSize = sequencesRows * sequencesCols * sizeof(char4);
    size_t scoresSize = length * sizeof(int);
    size_t indexesSize = length * sizeof(int);

    size_t memory = offsetsSize + 2 * lengthsSize + sequencesSize + 
        indexesSize + scoresSize + hBusSize;

    return memory;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// DATABASE SCORING

extern void scoreShortDatabaseGpu(int* scores, int type, Chain* query, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, &query, 1, shortDatabase, scorer, indexes, 
        indexesLen, cards, cardsLen, thread);
}

extern void scoreShortDatabasesGpu(int* scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, queries, queriesLen, shortDatabase, scorer,
        indexes, indexesLen, cards, cardsLen, thread);
}

//------------------------------------------------------------------------------

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR 

static ShortDatabase* createDatabase(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen) {
    
    ASSERT(cardsLen > 0, "no GPUs available");

    //**************************************************************************
    // FILTER DATABASE AND REMEBER ORDER
    
    int length = 0;
    
    for (int i = 0; i < databaseLen; ++i) {
    
        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            length++;
        }
    }
    
    if (length == 0) {
        return NULL;
    }
    
    int2* orderPacked = (int2*) malloc(length * sizeof(int2));

    for (int i = 0, j = 0; i < databaseLen; ++i) {
    
        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            orderPacked[j].x = i;
            orderPacked[j].y = n;
            j++;
        }
    }
    
    qsort(orderPacked, length, sizeof(int2), int2CmpY);
    
    LOG("Short database length: %d", length);

    //**************************************************************************

    //**************************************************************************
    // CALCULATE GRID DIMENSIONS
    
    int sequencesCols = THREADS * BLOCKS;
    int sequencesRows = 0;

    int blocks = 0;
    for (int i = sequencesCols - 1; i < length; i += sequencesCols) {
        int n = chainGetLength(database[orderPacked[i].x]);
        sequencesRows += (n >> 2) + ((n & 3) > 0);
        blocks++;
    }
    
    if (length % sequencesCols != 0) {
        int n = chainGetLength(database[orderPacked[length - 1].x]);
        sequencesRows += (n >> 2) + ((n & 3) > 0);
        blocks++;
    }
    
    LOG("Short database grid: %d(%d)x%d", sequencesRows, blocks, sequencesCols);
    
    //**************************************************************************
    
    //**************************************************************************
    // INIT STRUCTURES
    
    size_t offsetsSize = blocks * sizeof(int);
    int* offsets = (int*) malloc(offsetsSize);
    
    size_t lengthsSize = blocks * sequencesCols * sizeof(int);
    int* lengths = (int*) malloc(lengthsSize);
    int* lengthsPadded = (int*) malloc(lengthsSize);
    
    size_t sequencesSize = sequencesRows * sequencesCols * sizeof(char4);
    char4* sequences = (char4*) malloc(sequencesSize);
    
    //***********f***************************************************************

    //**************************************************************************
    // CREATE GRID
    
    // tmp
    size_t sequenceSize = chainGetLength(database[orderPacked[length - 1].x]) + 4;
    char* sequence = (char*) malloc(sequenceSize);

    offsets[0] = 0;
    for(int i = 0, j = 0, cx = 0, cy = 0; i < length; i++){

        //get the sequence and its length
        Chain* chain = database[orderPacked[i].x];
        int n = chainGetLength(chain);    
        
        lengths[j * sequencesCols + cx] = n;
        
        chainCopyCodes(chain, sequence);
        memset(sequence + n, 127, 4 * sizeof(char));

        int n4 = (n >> 2) + ((n & 3) > 0);

        lengthsPadded[j * sequencesCols + cx] = n4;
        
        char4* ptr = sequences + cy * sequencesCols + cx;
        for(int k = 0; k < n; k += 4){
            ptr->x = sequence[k];
            ptr->y = sequence[k + 1];
            ptr->z = sequence[k + 2];
            ptr->w = sequence[k + 3];
            ptr += sequencesCols;
        }

        cx++;
        
        if(cx == sequencesCols){
            offsets[j + 1] = offsets[j] + n4;
            cx = 0;
            cy += n4;
            j++;
        }
    }
    
    free(sequence);
    
    //**************************************************************************
    
    //**************************************************************************
    // CREATE POSITION ARRAY
    
    int* positions = (int*) malloc(databaseLen * sizeof(int));

    for (int i = 0; i < databaseLen; ++i) {
        positions[i] = -1;
    }
    
    for (int i = 0; i < length; ++i) {
        positions[orderPacked[i].x] = i;
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CREATE ORDER ARRAY
    
    size_t orderSize = length * sizeof(int);
    int* order = (int*) malloc(orderSize);

    for (int i = 0; i < length; ++i) {
        order[i] = orderPacked[i].x;
    }
     
    //**************************************************************************
    
    //**************************************************************************
    // CREATE DEFAULT INDEXES
    
    size_t indexesSize = length * sizeof(int);
    int* indexes = (int*) malloc(indexesSize);

    for (int i = 0; i < length; ++i) {
        indexes[i] = i;
    }
     
    //**************************************************************************

    //**************************************************************************
    // CREATE GPU DATABASES
    
    size_t gpuDatabasesSize = cardsLen * sizeof(GpuDatabase);
    GpuDatabase* gpuDatabases = (GpuDatabase*) malloc(gpuDatabasesSize);

    GpuDatabaseContext* contexts = 
        (GpuDatabaseContext*) malloc(cardsLen * sizeof(GpuDatabaseContext));

    Thread* threads = (Thread*) malloc(cardsLen * sizeof(Thread));

    for (int i = 0; i < cardsLen; ++i) {

        GpuDatabaseContext* context = &(contexts[i]);

        context->card = cards[i];
        context->length = length;
        context->blocks = blocks;
        context->offsets = offsets;
        context->offsetsSize = offsetsSize;
        context->lengths = lengths;
        context->lengthsPadded = lengthsPadded;
        context->lengthsSize = lengthsSize;
        context->sequences = sequences;
        context->sequencesCols = sequencesCols;
        context->sequencesRows = sequencesRows;
        context->sequencesSize = sequencesSize;
        context->indexes = indexes;
        context->indexesSize = indexesSize;
        context->gpuDatabase = gpuDatabases + i;
    }

    for (int i = 1; i < cardsLen; ++i) {
        threadCreate(&(threads[i]), createDatabaseGpu, (void*) &(contexts[i]));
    }

    createDatabaseGpu((void*) &(contexts[0]));

    for (int i = 1; i < cardsLen; ++i) {
        threadJoin(threads[i]);
    }

    free(contexts);
    free(threads);

    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    free(orderPacked);
    free(offsets);
    free(lengths);
    free(lengthsPadded);
    free(sequences);

    //**************************************************************************
    
    size_t shortDatabaseSize = sizeof(struct ShortDatabase);
    ShortDatabase* shortDatabase = (ShortDatabase*) malloc(shortDatabaseSize);
    
    shortDatabase->database = database;
    shortDatabase->databaseLen = databaseLen;
    shortDatabase->length = length;
    shortDatabase->positions = positions;
    shortDatabase->order = order;
    shortDatabase->indexes = indexes;
    shortDatabase->blocks = blocks;
    shortDatabase->sequencesRows = sequencesRows;
    shortDatabase->sequencesCols = sequencesCols;
    shortDatabase->gpuDatabases = gpuDatabases;
    shortDatabase->gpuDatabasesLen = cardsLen;
    
    return shortDatabase;
}

static void* createDatabaseGpu(void* param) {

    GpuDatabaseContext* context = (GpuDatabaseContext*) param;

    int card = context->card;
    int length = context->length;
    int blocks = context->blocks;
    int* offsets = context->offsets;
    size_t offsetsSize = context->offsetsSize;
    int* lengths = context->lengths;
    int* lengthsPadded = context->lengthsPadded;
    size_t lengthsSize = context->lengthsSize;
    char4* sequences = context->sequences;
    int sequencesCols = context->sequencesCols;
    int sequencesRows = context->sequencesRows;
    size_t sequencesSize = context->sequencesSize;
    int* indexes = context->indexes;
    size_t indexesSize = context->indexesSize;
    GpuDatabase* gpuDatabase = context->gpuDatabase;

    CUDA_SAFE_CALL(cudaSetDevice(card));

    int* offsetsGpu;
    CUDA_SAFE_CALL(cudaMalloc(&offsetsGpu, offsetsSize));
    CUDA_SAFE_CALL(cudaMemcpy(offsetsGpu, offsets, offsetsSize, TO_GPU));
    
    int* lengthsGpu;
    CUDA_SAFE_CALL(cudaMalloc(&lengthsGpu, lengthsSize));
    CUDA_SAFE_CALL(cudaMemcpy(lengthsGpu, lengths, lengthsSize, TO_GPU));

    int* lengthsPaddedGpu;
    CUDA_SAFE_CALL(cudaMalloc(&lengthsPaddedGpu, lengthsSize));
    CUDA_SAFE_CALL(cudaMemcpy(lengthsPaddedGpu, lengthsPadded, lengthsSize, TO_GPU));
    
    cudaArray* sequencesGpu;
    cudaChannelFormatDesc channel = seqsTexture.channelDesc;
    CUDA_SAFE_CALL(cudaMallocArray(&sequencesGpu, &channel, sequencesCols, sequencesRows)); 
    CUDA_SAFE_CALL(cudaMemcpyToArray(sequencesGpu, 0, 0, sequences, sequencesSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTextureToArray(seqsTexture, sequencesGpu));

    int* indexesGpu;
    CUDA_SAFE_CALL(cudaMalloc(&indexesGpu, indexesSize));
    CUDA_SAFE_CALL(cudaMemcpy(indexesGpu, indexes, indexesSize, TO_GPU));
    
    // additional structures

    size_t scoresSize = length * sizeof(int);
    int* scoresGpu;
    CUDA_SAFE_CALL(cudaMalloc(&scoresGpu, scoresSize));

    int2* hBusGpu;
    int hBusHeight = (sequencesRows - offsets[blocks - 1]) * 4;
    size_t hBusSize = sequencesCols * hBusHeight * sizeof(int2);
    CUDA_SAFE_CALL(cudaMalloc(&hBusGpu, hBusSize));

    gpuDatabase->card = card;
    gpuDatabase->offsets = offsetsGpu;
    gpuDatabase->lengths = lengthsGpu;
    gpuDatabase->lengthsPadded = lengthsPaddedGpu;
    gpuDatabase->sequences = sequencesGpu;
    gpuDatabase->indexes = indexesGpu;
    gpuDatabase->scores = scoresGpu;
    gpuDatabase->hBus = hBusGpu;
    
#ifdef DEBUG
    size_t memory = offsetsSize + 2 * lengthsSize + sequencesSize + 
        indexesSize + scoresSize + hBusSize;

    LOG("Short database using %.2lfMBs on card %d", memory / 1024.0 / 1024.0, card);
#endif

    return NULL;
}

static void deleteDatabase(ShortDatabase* database) {

    if (database == NULL) {
        return;
    }
    
    for (int i = 0; i < database->gpuDatabasesLen; ++i) {
    
        GpuDatabase* gpuDatabase = &(database->gpuDatabases[i]);
        
        CUDA_SAFE_CALL(cudaSetDevice(gpuDatabase->card));

        CUDA_SAFE_CALL(cudaFree(gpuDatabase->offsets));
        CUDA_SAFE_CALL(cudaFree(gpuDatabase->lengths));
        CUDA_SAFE_CALL(cudaFree(gpuDatabase->lengthsPadded));
        CUDA_SAFE_CALL(cudaFreeArray(gpuDatabase->sequences));
        CUDA_SAFE_CALL(cudaFree(gpuDatabase->indexes));
        CUDA_SAFE_CALL(cudaFree(gpuDatabase->scores));
        CUDA_SAFE_CALL(cudaFree(gpuDatabase->hBus));

        CUDA_SAFE_CALL(cudaUnbindTexture(seqsTexture));
    }

    free(database->gpuDatabases);
    free(database->positions);
    free(database->order);
    free(database->indexes);

    free(database);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// ENTRY 

static void scoreDatabase(int* scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    
    ASSERT(cardsLen > 0, "no GPUs available");
    
    Context* param = (Context*) malloc(sizeof(Context));
    
    param->scores = scores;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->shortDatabase = shortDatabase;
    param->scorer = scorer;
    param->indexes = indexes;
    param->indexesLen = indexesLen;
    param->cards = cards;
    param->cardsLen = cardsLen;
    
    if (thread == NULL) {
        scoreDatabaseThread(param);
    } else {
        threadCreate(thread, scoreDatabaseThread, (void*) param);
    }
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// DATABASE SCORING

static void* scoreDatabaseThread(void* param) {

    Context* context = (Context*) param;
    
    int* scores = context->scores;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    ShortDatabase* shortDatabase = context->shortDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    if (shortDatabase == NULL) {
        return NULL;
    }

    //**************************************************************************
    // CREATE NEW INDEXES ARRAY IF NEEDED
    
    int* newIndexes = NULL;
    int newIndexesLen = 0;

    int deleteIndexes;

    if (indexes != NULL) {

        // translate and filter indexes, also make sure that indexes are 
        // sorted by size 
    
        int length = shortDatabase->length;
        int databaseLen = shortDatabase->databaseLen;
        int* positions = shortDatabase->positions;
        
        char* solveMask = (char*) malloc(length * sizeof(char));
        memset(solveMask, 0, length);
        
        newIndexesLen = 0;
        for (int i = 0; i < indexesLen; ++i) {
            
            int idx = indexes[i];
            if (idx < 0 || idx > databaseLen || positions[idx] == -1) {
                continue;
            }
            
            solveMask[positions[idx]] = 1;
            newIndexesLen++;
        }
        
        newIndexes = (int*) malloc(newIndexesLen * sizeof(int));
        
        for (int i = 0, j = 0; i < length; ++i) {
            if (solveMask[i]) {
                newIndexes[j++] = i;
            }
        }
        
        free(solveMask);

        deleteIndexes = 1;

    } else {
        // load prebuilt defaults
        newIndexes = shortDatabase->indexes;
        newIndexesLen = shortDatabase->length;
        deleteIndexes = 0;
    }
    
    //**************************************************************************

    //**************************************************************************
    // CHOOSE SOLVING FUNCTION
    
    ScoringFunction function;
    switch (type) {
    case SW_ALIGN: 
        function = swSolveShortGpu;
        break;
    case NW_ALIGN: 
        function = nwSolveShortGpu;
        break;
    case HW_ALIGN:
        function = hwSolveShortGpu;
        break;
    case OV_ALIGN:
        function = ovSolveShortGpu;
        break;
    default:
        ERROR("Wrong align type");
    }
    
    //**************************************************************************

    //**************************************************************************
    // SCORE MULTITHREADED

    if (queriesLen < cardsLen) {
        scoreDatabaseMulti(scores, type, function, queries, queriesLen, 
            shortDatabase, scorer, newIndexes, newIndexesLen, cards, cardsLen);
    } else {
        scoreDatabaseSingle(scores, type, function, queries, queriesLen, 
            shortDatabase, scorer, newIndexes, newIndexesLen, cards, cardsLen);
    }
    
    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    if (deleteIndexes) {
        free(newIndexes);
    }

    free(param);
    
    //**************************************************************************
    
    return NULL;
}

static void scoreDatabaseMulti(int* scores, int type, 
    ScoringFunction scoringFunction, Chain** queries, int queriesLen, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen) {
    
    //**************************************************************************
    // CREATE QUERY PROFILES
    
    size_t profilesSize = queriesLen * sizeof(QueryProfile*);
    QueryProfile** profiles = (QueryProfile**) malloc(profilesSize);
    
    for (int i = 0; i < queriesLen; ++i) {
        profiles[i] = createQueryProfile(queries[i], scorer);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CREATE BALANCING DATA

    Chain** database = shortDatabase->database;
    int* order = shortDatabase->order;
    int sequencesCols = shortDatabase->sequencesCols;
    
    int blocks = indexesLen / sequencesCols + ((indexesLen % sequencesCols) > 0);

    size_t weightsSize = blocks * sizeof(int);
    int* weights = (int*) malloc(weightsSize);
    memset(weights, 0, weightsSize);

    for (int i = 0, j = 0; i < indexesLen; ++i) {

        weights[j] += chainGetLength(database[order[indexes[i]]]);

        if ((i + 1) % sequencesCols == 0) {
            j++;
        }
    }

    //**************************************************************************

    //**************************************************************************
    // SCORE MULTICARDED
    
    int contextsLen = cardsLen * queriesLen;
    size_t contextsSize = contextsLen * sizeof(KernelContext);
    KernelContext* contexts = (KernelContext*) malloc(contextsSize);
    
    size_t tasksSize = contextsLen * sizeof(Thread);
    Thread* tasks = (Thread*) malloc(tasksSize);

    int databaseLen = shortDatabase->databaseLen;
    
    int cardsChunk = cardsLen / queriesLen;
    int cardsAdd = cardsLen % queriesLen;
    int cardsOff = 0;

    int* idxChunksOff = (int*) malloc(cardsLen * sizeof(int));
    int* idxChunksLens = (int*) malloc(cardsLen * sizeof(int));
    int idxChunksLen = 0;
    int idxLastFix = (sequencesCols - indexesLen % sequencesCols) % sequencesCols;

    int length = 0;

    for (int i = 0, k = 0; i < queriesLen; ++i) {

        int cCardsLen = cardsChunk + (i < cardsAdd);
        int* cCards = cards + cardsOff;
        cardsOff += cCardsLen;
        
        QueryProfile* queryProfile = profiles[i];

        int chunks = min(cCardsLen, blocks);
        if (chunks != idxChunksLen) {
            weightChunkArray(idxChunksOff, idxChunksLens, &idxChunksLen, 
                weights, blocks, chunks);
        }
        
        for (int j = 0; j < idxChunksLen; ++j, ++k) {
        
            int off = idxChunksOff[j] * sequencesCols;
            int len = idxChunksLens[j] * sequencesCols;
            if (j == idxChunksLen - 1) {
                len -= idxLastFix;
            }
            
            contexts[k].scores = scores + i * databaseLen;
            contexts[k].type = type;
            contexts[k].scoringFunction = scoringFunction;
            contexts[k].queryProfile = queryProfile;
            contexts[k].query = queries[i];
            contexts[k].shortDatabase = shortDatabase;
            contexts[k].scorer = scorer;
            contexts[k].indexes = indexes + off;
            contexts[k].indexesLen = len;
            contexts[k].card = cCards[j];

            threadCreate(&(tasks[k]), kernelThread, &(contexts[k]));
            length++;
        }
    }
    
    for (int i = 0; i < length; ++i) {
        threadJoin(tasks[i]);
    }

    free(tasks);
    free(contexts);

    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    for (int i = 0; i < queriesLen; ++i) {
        deleteQueryProfile(profiles[i]);
    }

    free(profiles);
    free(weights);
    free(idxChunksOff);
    free(idxChunksLens);
    
    //**************************************************************************
}

static void scoreDatabaseSingle(int* scores, int type, 
    ScoringFunction scoringFunction, Chain** queries, int queriesLen, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen) {

    //**************************************************************************
    // CREATE CONTEXTS
    
    size_t contextsSize = cardsLen * sizeof(KernelContext);
    KernelContexts* contexts = (KernelContexts*) malloc(contextsSize);
    
    for (int i = 0; i < cardsLen; ++i) {
        size_t size = queriesLen * sizeof(KernelContext);
        contexts[i].contexts = (KernelContext*) malloc(size);
        contexts[i].contextsLen = 0;
        contexts[i].cells = 0;
    }
    
    //**************************************************************************    
    
    //**************************************************************************
    // SCORE MULTITHREADED
    
    size_t tasksSize = cardsLen * sizeof(Thread);
    Thread* tasks = (Thread*) malloc(tasksSize);
    
    int databaseLen = shortDatabase->databaseLen;
    
    // balance tasks by round roobin, cardsLen is pretty small (CUDA cards)
    for (int i = 0; i < queriesLen; ++i) {
        
        int minIdx = 0;
        long long minVal = contexts[0].cells;
        for (int j = 1; j < cardsLen; ++j) {
            if (contexts[j].cells < minVal) {
                minVal = contexts[j].cells;
                minIdx = j;
            }
        }
        
        KernelContext context;
        context.scores = scores + i * databaseLen;
        context.type = type;
        context.scoringFunction = scoringFunction;
        context.queryProfile = NULL;
        context.query = queries[i];
        context.shortDatabase = shortDatabase;
        context.scorer = scorer;
        context.indexes = indexes;
        context.indexesLen = indexesLen;
        context.card = cards[minIdx];

        contexts[minIdx].contexts[contexts[minIdx].contextsLen++] = context;
        contexts[minIdx].cells += chainGetLength(queries[i]);
    }
    
    for (int i = 0; i < cardsLen; ++i) {
        threadCreate(&(tasks[i]), kernelsThread, &(contexts[i]));
    }

    for (int i = 0; i < cardsLen; ++i) {
        threadJoin(tasks[i]);
    }
    free(tasks);

    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    for (int i = 0; i < cardsLen; ++i) {
        free(contexts[i].contexts);
    }
    free(contexts);

    //**************************************************************************
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void* kernelsThread(void* param) {

    KernelContexts* context = (KernelContexts*) param;

    KernelContext* contexts = context->contexts;
    int contextsLen = context->contextsLen;

    for (int i = 0; i < contextsLen; ++i) {
    
        Chain* query = contexts[i].query;
        Scorer* scorer = contexts[i].scorer;
        int card = contexts[i].card;
        
        int currentCard;
        CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
        if (currentCard != card) {
            CUDA_SAFE_CALL(cudaSetDevice(card));
        }
    
        contexts[i].queryProfile = createQueryProfile(query, scorer);
        
        kernelThread(&(contexts[i]));
        
        deleteQueryProfile(contexts[i].queryProfile);
    }
    
    return NULL;
}

static void* kernelThread(void* param) {

    KernelContext* context = (KernelContext*) param;
    
    int* scores = context->scores;
    int type = context->type;
    ScoringFunction scoringFunction = context->scoringFunction;
    Chain* query = context->query;
    QueryProfile* queryProfile = context->queryProfile;
    ShortDatabase* shortDatabase = context->shortDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int card = context->card;
    
    //**************************************************************************
    // FIND DATABASE
    
    GpuDatabase* gpuDatabases = shortDatabase->gpuDatabases;
    int gpuDatabasesLen = shortDatabase->gpuDatabasesLen;
    
    GpuDatabase* gpuDatabase = NULL;
    
    for (int i = 0; i < gpuDatabasesLen; ++i) {
        if (gpuDatabases[i].card == card) {
            gpuDatabase = &(gpuDatabases[i]);
            break;
        }
    }

    ASSERT(gpuDatabase != NULL, "Short database not available on card %d", card);

    //**************************************************************************
    
    //**************************************************************************
    // CUDA SETUP
    
    int currentCard;
    CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
    if (currentCard != card) {
        CUDA_SAFE_CALL(cudaSetDevice(card));
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // FIX INDEXES
    
    int deleteIndexes;
    int* indexesGpu;
    
    if (indexesLen == shortDatabase->length) {
        indexes = shortDatabase->indexes;
        indexesLen = shortDatabase->length;
        indexesGpu = gpuDatabase->indexes;
        deleteIndexes = 0;
    } else {
        size_t indexesSize = indexesLen * sizeof(int);
        CUDA_SAFE_CALL(cudaMalloc(&indexesGpu, indexesSize));
        CUDA_SAFE_CALL(cudaMemcpy(indexesGpu, indexes, indexesSize, TO_GPU));
        deleteIndexes = 1;
    }

    //**************************************************************************
    
    //**************************************************************************
    // PREPARE GPU
    
    QueryProfileGpu* queryProfileGpu = createQueryProfileGpu(queryProfile);
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int rows = queryProfile->length;
    int rowsGpu = queryProfile->height * 4;
    int sequencesCols = shortDatabase->sequencesCols;
    
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapOpen_, &gapOpen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapExtend_, &gapExtend, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rows_, &rows, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rowsPadded_, &rowsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(width_, &sequencesCols, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(length_, &indexesLen, sizeof(int)));
    
    //**************************************************************************

    //**************************************************************************
    // PREPARE CPU

    int lastIndexSolvedCpu = 0;
    int firstIndexSolvedGpu = INT_MAX;

    Mutex indexSolvedMutex;
    mutexCreate(&indexSolvedMutex);
    
    KernelContextCpu* contextCpu = (KernelContextCpu*) malloc(sizeof(KernelContextCpu));

    contextCpu->scores = scores;
    contextCpu->type = type;
    contextCpu->query = query;
    contextCpu->shortDatabase = shortDatabase;
    contextCpu->scorer = scorer;
    contextCpu->indexes = indexes;
    contextCpu->indexesLen = indexesLen;
    contextCpu->lastIndexSolvedCpu = &lastIndexSolvedCpu;
    contextCpu->firstIndexSolvedGpu = &firstIndexSolvedGpu;
    contextCpu->indexSolvedMutex = &indexSolvedMutex;
    
    //**************************************************************************

    //**************************************************************************
    // SOLVE

    Thread thread;
    threadCreate(&thread, kernelThreadCpu, contextCpu);

    int blocks = shortDatabase->blocks;
    
    int* offsetsGpu = gpuDatabase->offsets;
    int* lengthsGpu = gpuDatabase->lengths;
    int* lengthsPaddedGpu = gpuDatabase->lengthsPadded;
    int* scoresGpu = gpuDatabase->scores;
    int2* hBusGpu = gpuDatabase->hBus;
    
    TIMER_START("Short GPU solving: %d", indexesLen);

    for (int i = blocks - 1; i >= 0; --i) {

        if (sequencesCols * i > indexesLen) {
            continue;
        }

        // wait for iteration to finish
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        int firstIdx = sequencesCols * i;
        int lastIdx = min(sequencesCols * (i + 1) - 1, indexesLen - 1);

        // multithreaded, chech mutexes
        mutexLock(&indexSolvedMutex);

        // indexes already solved
        if (lastIdx < lastIndexSolvedCpu) {
            mutexUnlock(&indexSolvedMutex);
            break;
        }

        firstIndexSolvedGpu = min(firstIdx, firstIndexSolvedGpu);

        mutexUnlock(&indexSolvedMutex);

        scoringFunction<<<BLOCKS, THREADS>>>(scoresGpu, hBusGpu, lengthsGpu, 
            lengthsPaddedGpu, offsetsGpu, indexesGpu, i);
    }

    TIMER_STOP;

    threadJoin(thread);

    //**************************************************************************
    
    //**************************************************************************
    // SAVE RESULTS

    int length = shortDatabase->length;
    
    size_t scoresSize = length * sizeof(int);
    int* scoresCpu = (int*) malloc(scoresSize);

    CUDA_SAFE_CALL(cudaMemcpy(scoresCpu, scoresGpu, scoresSize, FROM_GPU));

    int* order = shortDatabase->order;
    
    for (int i = firstIndexSolvedGpu; i < indexesLen; ++i) {
        scores[order[indexes[i]]] = scoresCpu[indexes[i]];
    }
    
    free(scoresCpu);
                
    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY
    
    deleteQueryProfileGpu(queryProfileGpu);
    
    if (deleteIndexes) {
        CUDA_SAFE_CALL(cudaFree(indexesGpu));
    }

    mutexDelete(&indexSolvedMutex);
    free(contextCpu);

    //**************************************************************************
    
    return NULL;
}

static void* kernelThreadCpu(void* param) {

    int i;

    KernelContextCpu* context = (KernelContextCpu*) param;

    int* scores = context->scores;
    int type = context->type;
    Chain* query = context->query;
    ShortDatabase* shortDatabase = context->shortDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int* lastIndexSolvedCpu = context->lastIndexSolvedCpu;
    int* firstIndexSolvedGpu = context->firstIndexSolvedGpu;
    Mutex* indexSolvedMutex = context->indexSolvedMutex;

    int* order = shortDatabase->order;

    TIMER_START("Short CPU solving");

    //**************************************************************************
    // CREATE DATABASE
    

    Chain** database = (Chain**) malloc(indexesLen * sizeof(Chain*));
    int databaseLen = 0;

    for (i = 0; i < indexesLen; ++i) {

        Chain* chain = shortDatabase->database[order[indexes[i]]];

        if (chainGetLength(chain) > MAX_CPU_LEN) {
            break;
        }

        database[i] = chain;
        databaseLen++;
    }

    LOG("Max CPU chains: %d", databaseLen);

    //**************************************************************************

    //**************************************************************************
    // SOLVE

    int* scoresCpu = (int*) malloc(databaseLen * sizeof(int));

    int workers = min(CPU_WORKERS, databaseLen);

    CpuWorkerContext* contexts = (CpuWorkerContext*) malloc(workers * sizeof(CpuWorkerContext));
    Thread* tasks = (Thread*) malloc(workers * sizeof(Thread));

    for (i = 0; i < workers; ++i) {

        contexts[i].scores = scoresCpu;
        contexts[i].type = type;
        contexts[i].query = query;
        contexts[i].database = database;
        contexts[i].databaseLen = databaseLen;
        contexts[i].scorer = scorer;
        contexts[i].lastIndexSolvedCpu = lastIndexSolvedCpu;
        contexts[i].firstIndexSolvedGpu = firstIndexSolvedGpu;
        contexts[i].indexSolvedMutex = indexSolvedMutex;

        threadCreate(&(tasks[i]), cpuWorker, &(contexts[i]));
    }
    
    for (i = 0; i < workers; ++i) {
        threadJoin(tasks[i]);
    }

    free(tasks);
    free(contexts);

    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    LOG("CPU solved %d chains", *lastIndexSolvedCpu);

    for (int i = 0; i <= *lastIndexSolvedCpu; ++i) {
        scores[order[indexes[i]]] = scoresCpu[i];
    }
    
    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(scoresCpu);
    free(database);

    //**************************************************************************

    TIMER_STOP;

    return NULL;
}

static void* cpuWorker(void* param) {

    CpuWorkerContext* context = (CpuWorkerContext*) param;

    int* scores = context->scores;
    int type = context->type;
    Chain* query = context->query;
    Chain** database = context->database;
    int databaseLen = context->databaseLen;
    Scorer* scorer = context->scorer;
    int* lastIndexSolvedCpu = context->lastIndexSolvedCpu;
    int* firstIndexSolvedGpu = context->firstIndexSolvedGpu;
    Mutex* indexSolvedMutex = context->indexSolvedMutex;

    while (1) {

        mutexLock(indexSolvedMutex);

        int start = max(0, *lastIndexSolvedCpu);
        int length = min(CPU_WORKER_STEP, databaseLen - start);

        if (start >= databaseLen || start > *firstIndexSolvedGpu - THREADS * BLOCKS) {
            mutexUnlock(indexSolvedMutex);
            break;
        }

        *lastIndexSolvedCpu += length;

        mutexUnlock(indexSolvedMutex);

        scoreDatabaseCpu(scores + start, type, query, database + start, length, scorer);
    }

    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

__device__ static int gap(int index) {
    return (-gapOpen_ - index * gapExtend_) * (index >= 0);
}

__global__ static void hwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid + block * width_ >= length_) {
        return;
    }
    
    int id = indexes[tid + block * width_];
    int cols = lengthsPadded[id];
    int realCols = lengths[id];
    
    int colOff = id % width_;
    int rowOff = offsets[id / width_];
    
    int score = SCORE_MIN;
    
    int4 scrUp;
    int4 affUp;
    int4 mchUp;
    
    int4 scrDown;
    int4 affDown;
    int4 mchDown;
    
    int2 wBus;
    int del;
    
    int lastRow = rows_ - 1;
    
    for (int j = 0; j < cols * 4; ++j) {
        hBus[j * width_ + tid] = make_int2(0, SCORE_MIN);
    }
    
    for (int i = 0; i < rowsPadded_; i += 8) {
    
        scrUp = make_int4(gap(i), gap(i + 1), gap(i + 2), gap(i + 3));
        affUp = INT4_SCORE_MIN;
        mchUp = make_int4(gap(i - 1), gap(i), gap(i + 1), gap(i + 2));
        
        scrDown = make_int4(gap(i + 4), gap(i + 5), gap(i + 6), gap(i + 7));
        affDown = INT4_SCORE_MIN;
        mchDown = make_int4(gap(i + 3), gap(i + 4), gap(i + 5), gap(i + 6));
        
        for (int j = 0; j < cols; ++j) {
        
            int columnCodes = tex2D(seqsTexture, colOff, j + rowOff);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
            
                int validCol = (j * 4 + k) < realCols;
                
                wBus = hBus[(j * 4 + k) * width_ + tid];
                
                char code = (columnCodes >> (k << 3));
                char4 rowScores = tex2D(qpTexture, code, i / 4);
                
                del = max(wBus.x - gapOpen_, wBus.y - gapExtend_);
                affUp.x = max(scrUp.x - gapOpen_, affUp.x - gapExtend_);
                scrUp.x = mchUp.x + rowScores.x; 
                scrUp.x = max(scrUp.x, del);
                scrUp.x = max(scrUp.x, affUp.x);
                mchUp.x = wBus.x;
                if (i + 0 == lastRow && validCol) score = max(score, scrUp.x);
                
                del = max(scrUp.x - gapOpen_, del - gapExtend_);
                affUp.y = max(scrUp.y - gapOpen_, affUp.y - gapExtend_);
                scrUp.y = mchUp.y + rowScores.y; 
                scrUp.y = max(scrUp.y, del);
                scrUp.y = max(scrUp.y, affUp.y);
                mchUp.y = scrUp.x;
                if (i + 1 == lastRow && validCol) score = max(score, scrUp.y);
                
                del = max(scrUp.y - gapOpen_, del - gapExtend_);
                affUp.z = max(scrUp.z - gapOpen_, affUp.z - gapExtend_);
                scrUp.z = mchUp.z + rowScores.z; 
                scrUp.z = max(scrUp.z, del);
                scrUp.z = max(scrUp.z, affUp.z);
                mchUp.z = scrUp.y;
                if (i + 2 == lastRow && validCol) score = max(score, scrUp.z);
                
                del = max(scrUp.z - gapOpen_, del - gapExtend_);
                affUp.w = max(scrUp.w - gapOpen_, affUp.w - gapExtend_);
                scrUp.w = mchUp.w + rowScores.w; 
                scrUp.w = max(scrUp.w, del);
                scrUp.w = max(scrUp.w, affUp.w);
                mchUp.w = scrUp.z;
                if (i + 3 == lastRow && validCol) score = max(score, scrUp.w);

                rowScores = tex2D(qpTexture, code, i / 4 + 1);
                
                del = max(scrUp.w - gapOpen_, del - gapExtend_);
                affDown.x = max(scrDown.x - gapOpen_, affDown.x - gapExtend_);
                scrDown.x = mchDown.x + rowScores.x; 
                scrDown.x = max(scrDown.x, del);
                scrDown.x = max(scrDown.x, affDown.x);
                mchDown.x = scrUp.w;
                if (i + 4 == lastRow && validCol) score = max(score, scrDown.x);
                
                del = max(scrDown.x - gapOpen_, del - gapExtend_);
                affDown.y = max(scrDown.y - gapOpen_, affDown.y - gapExtend_);
                scrDown.y = mchDown.y + rowScores.y; 
                scrDown.y = max(scrDown.y, del);
                scrDown.y = max(scrDown.y, affDown.y);
                mchDown.y = scrDown.x;
                if (i + 5 == lastRow && validCol) score = max(score, scrDown.y);
                
                del = max(scrDown.y - gapOpen_, del - gapExtend_);
                affDown.z = max(scrDown.z - gapOpen_, affDown.z - gapExtend_);
                scrDown.z = mchDown.z + rowScores.z; 
                scrDown.z = max(scrDown.z, del);
                scrDown.z = max(scrDown.z, affDown.z);
                mchDown.z = scrDown.y;
                if (i + 6 == lastRow && validCol) score = max(score, scrDown.z);
                
                del = max(scrDown.z - gapOpen_, del - gapExtend_);
                affDown.w = max(scrDown.w - gapOpen_, affDown.w - gapExtend_);
                scrDown.w = mchDown.w + rowScores.w; 
                scrDown.w = max(scrDown.w, del);
                scrDown.w = max(scrDown.w, affDown.w);
                mchDown.w = scrDown.z;
                if (i + 7 == lastRow && validCol) score = max(score, scrDown.w);
                
                wBus.x = scrDown.w;
                wBus.y = del;
                
                hBus[(j * 4 + k) * width_ + tid] = wBus;
            }
        }
    }
    
    scores[id] = score;
}

__global__ static void nwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid + block * width_ >= length_) {
        return;
    }
    
    int id = indexes[tid + block * width_];
    int cols = lengthsPadded[id];
    int realCols = lengths[id];
    
    int colOff = id % width_;
    int rowOff = offsets[id / width_];
    
    int score = SCORE_MIN;
    
    int4 scrUp;
    int4 affUp;
    int4 mchUp;
    
    int4 scrDown;
    int4 affDown;
    int4 mchDown;
    
    int2 wBus;
    int del;
    
    int lastRow = rows_ - 1;

    for (int j = 0; j < cols * 4; ++j) {
        hBus[j * width_ + tid] = make_int2(gap(j), SCORE_MIN);
    }
    
    for (int i = 0; i < rowsPadded_; i += 8) {
    
        scrUp = make_int4(gap(i), gap(i + 1), gap(i + 2), gap(i + 3));
        affUp = INT4_SCORE_MIN;
        mchUp = make_int4(gap(i - 1), gap(i), gap(i + 1), gap(i + 2));
        
        scrDown = make_int4(gap(i + 4), gap(i + 5), gap(i + 6), gap(i + 7));
        affDown = INT4_SCORE_MIN;
        mchDown = make_int4(gap(i + 3), gap(i + 4), gap(i + 5), gap(i + 6));
        
        for (int j = 0; j < cols; ++j) {
        
            int columnCodes = tex2D(seqsTexture, colOff, j + rowOff);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
            
                int lastCol = (j * 4 + k) == (realCols - 1);
                
                wBus = hBus[(j * 4 + k) * width_ + tid];
                
                char code = (columnCodes >> (k << 3));
                char4 rowScores = tex2D(qpTexture, code, i / 4);
                
                del = max(wBus.x - gapOpen_, wBus.y - gapExtend_);
                affUp.x = max(scrUp.x - gapOpen_, affUp.x - gapExtend_);
                scrUp.x = mchUp.x + rowScores.x; 
                scrUp.x = max(scrUp.x, del);
                scrUp.x = max(scrUp.x, affUp.x);
                mchUp.x = wBus.x;
                if (i + 0 == lastRow && lastCol) score = scrUp.x;
                
                del = max(scrUp.x - gapOpen_, del - gapExtend_);
                affUp.y = max(scrUp.y - gapOpen_, affUp.y - gapExtend_);
                scrUp.y = mchUp.y + rowScores.y; 
                scrUp.y = max(scrUp.y, del);
                scrUp.y = max(scrUp.y, affUp.y);
                mchUp.y = scrUp.x;
                if (i + 1 == lastRow && lastCol) score = scrUp.y;
                
                del = max(scrUp.y - gapOpen_, del - gapExtend_);
                affUp.z = max(scrUp.z - gapOpen_, affUp.z - gapExtend_);
                scrUp.z = mchUp.z + rowScores.z; 
                scrUp.z = max(scrUp.z, del);
                scrUp.z = max(scrUp.z, affUp.z);
                mchUp.z = scrUp.y;
                if (i + 2 == lastRow && lastCol) score = scrUp.z;
                
                del = max(scrUp.z - gapOpen_, del - gapExtend_);
                affUp.w = max(scrUp.w - gapOpen_, affUp.w - gapExtend_);
                scrUp.w = mchUp.w + rowScores.w; 
                scrUp.w = max(scrUp.w, del);
                scrUp.w = max(scrUp.w, affUp.w);
                mchUp.w = scrUp.z;
                if (i + 3 == lastRow && lastCol) score = scrUp.w;

                rowScores = tex2D(qpTexture, code, i / 4 + 1);
                
                del = max(scrUp.w - gapOpen_, del - gapExtend_);
                affDown.x = max(scrDown.x - gapOpen_, affDown.x - gapExtend_);
                scrDown.x = mchDown.x + rowScores.x; 
                scrDown.x = max(scrDown.x, del);
                scrDown.x = max(scrDown.x, affDown.x);
                mchDown.x = scrUp.w;
                if (i + 4 == lastRow && lastCol) score = scrDown.x;
                
                del = max(scrDown.x - gapOpen_, del - gapExtend_);
                affDown.y = max(scrDown.y - gapOpen_, affDown.y - gapExtend_);
                scrDown.y = mchDown.y + rowScores.y; 
                scrDown.y = max(scrDown.y, del);
                scrDown.y = max(scrDown.y, affDown.y);
                mchDown.y = scrDown.x;
                if (i + 5 == lastRow && lastCol) score = scrDown.y;
                
                del = max(scrDown.y - gapOpen_, del - gapExtend_);
                affDown.z = max(scrDown.z - gapOpen_, affDown.z - gapExtend_);
                scrDown.z = mchDown.z + rowScores.z; 
                scrDown.z = max(scrDown.z, del);
                scrDown.z = max(scrDown.z, affDown.z);
                mchDown.z = scrDown.y;
                if (i + 6 == lastRow && lastCol) score = scrDown.z;
                
                del = max(scrDown.z - gapOpen_, del - gapExtend_);
                affDown.w = max(scrDown.w - gapOpen_, affDown.w - gapExtend_);
                scrDown.w = mchDown.w + rowScores.w; 
                scrDown.w = max(scrDown.w, del);
                scrDown.w = max(scrDown.w, affDown.w);
                mchDown.w = scrDown.z;
                if (i + 7 == lastRow && lastCol) score = scrDown.w;
                
                wBus.x = scrDown.w;
                wBus.y = del;
                
                hBus[(j * 4 + k) * width_ + tid] = wBus;
            }
        }
    }
    
    scores[id] = score;
}

__global__ static void ovSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid + block * width_ >= length_) {
        return;
    }
    
    int id = indexes[tid + block * width_];
    int cols = lengthsPadded[id];
    int realCols = lengths[id];
    
    int colOff = id % width_;
    int rowOff = offsets[id / width_];
    
    int score = SCORE_MIN;
    
    int4 scrUp;
    int4 affUp;
    int4 mchUp;
    
    int4 scrDown;
    int4 affDown;
    int4 mchDown;
    
    int2 wBus;
    int del;
    
    int lastRow = rows_ - 1;
    
    for (int j = 0; j < cols * 4; ++j) {
        hBus[j * width_ + tid] = make_int2(0, SCORE_MIN);
    }
    
    for (int i = 0; i < rowsPadded_; i += 8) {
    
        scrUp = INT4_ZERO;
        affUp = INT4_SCORE_MIN;
        mchUp = INT4_ZERO;
        
        scrDown = INT4_ZERO;
        affDown = INT4_SCORE_MIN;
        mchDown = INT4_ZERO;
        
        for (int j = 0; j < cols; ++j) {
        
            int columnCodes = tex2D(seqsTexture, colOff, j + rowOff);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
            
                int lastCol = (j * 4 + k) == (realCols - 1);
                
                wBus = hBus[(j * 4 + k) * width_ + tid];
                
                char code = (columnCodes >> (k << 3));
                char4 rowScores = tex2D(qpTexture, code, i / 4);
                
                del = max(wBus.x - gapOpen_, wBus.y - gapExtend_);
                affUp.x = max(scrUp.x - gapOpen_, affUp.x - gapExtend_);
                scrUp.x = mchUp.x + rowScores.x; 
                scrUp.x = max(scrUp.x, del);
                scrUp.x = max(scrUp.x, affUp.x);
                mchUp.x = wBus.x;
                if (i + 0 == lastRow || lastCol) score = max(score, scrUp.x);
                
                del = max(scrUp.x - gapOpen_, del - gapExtend_);
                affUp.y = max(scrUp.y - gapOpen_, affUp.y - gapExtend_);
                scrUp.y = mchUp.y + rowScores.y; 
                scrUp.y = max(scrUp.y, del);
                scrUp.y = max(scrUp.y, affUp.y);
                mchUp.y = scrUp.x;
                if (i + 1 == lastRow || lastCol) score = max(score, scrUp.y);
                
                del = max(scrUp.y - gapOpen_, del - gapExtend_);
                affUp.z = max(scrUp.z - gapOpen_, affUp.z - gapExtend_);
                scrUp.z = mchUp.z + rowScores.z; 
                scrUp.z = max(scrUp.z, del);
                scrUp.z = max(scrUp.z, affUp.z);
                mchUp.z = scrUp.y;
                if (i + 2 == lastRow || lastCol) score = max(score, scrUp.z);
                
                del = max(scrUp.z - gapOpen_, del - gapExtend_);
                affUp.w = max(scrUp.w - gapOpen_, affUp.w - gapExtend_);
                scrUp.w = mchUp.w + rowScores.w; 
                scrUp.w = max(scrUp.w, del);
                scrUp.w = max(scrUp.w, affUp.w);
                mchUp.w = scrUp.z;
                if (i + 3 == lastRow || lastCol) score = max(score, scrUp.w);

                rowScores = tex2D(qpTexture, code, i / 4 + 1);
                
                del = max(scrUp.w - gapOpen_, del - gapExtend_);
                affDown.x = max(scrDown.x - gapOpen_, affDown.x - gapExtend_);
                scrDown.x = mchDown.x + rowScores.x; 
                scrDown.x = max(scrDown.x, del);
                scrDown.x = max(scrDown.x, affDown.x);
                mchDown.x = scrUp.w;
                if (i + 4 == lastRow || lastCol) score = max(score, scrDown.x);
                
                del = max(scrDown.x - gapOpen_, del - gapExtend_);
                affDown.y = max(scrDown.y - gapOpen_, affDown.y - gapExtend_);
                scrDown.y = mchDown.y + rowScores.y; 
                scrDown.y = max(scrDown.y, del);
                scrDown.y = max(scrDown.y, affDown.y);
                mchDown.y = scrDown.x;
                if (i + 5 == lastRow || lastCol) score = max(score, scrDown.y);
                
                del = max(scrDown.y - gapOpen_, del - gapExtend_);
                affDown.z = max(scrDown.z - gapOpen_, affDown.z - gapExtend_);
                scrDown.z = mchDown.z + rowScores.z; 
                scrDown.z = max(scrDown.z, del);
                scrDown.z = max(scrDown.z, affDown.z);
                mchDown.z = scrDown.y;
                if (i + 6 == lastRow || lastCol) score = max(score, scrDown.z);
                
                del = max(scrDown.z - gapOpen_, del - gapExtend_);
                affDown.w = max(scrDown.w - gapOpen_, affDown.w - gapExtend_);
                scrDown.w = mchDown.w + rowScores.w; 
                scrDown.w = max(scrDown.w, del);
                scrDown.w = max(scrDown.w, affDown.w);
                mchDown.w = scrDown.z;
                if (i + 7 == lastRow || lastCol) score = max(score, scrDown.w);
                
                wBus.x = scrDown.w;
                wBus.y = del;
                
                hBus[(j * 4 + k) * width_ + tid] = wBus;
            }
        }
    }
    
    scores[id] = score;
}

__global__ static void swSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int* offsets, int* indexes, int block) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid + block * width_ >= length_) {
        return;
    }
    
    int id = indexes[tid + block * width_];
    int cols = lengthsPadded[id];
    
    int colOff = id % width_;
    int rowOff = offsets[id / width_];
    
    int score = 0;
    
    int4 scrUp;
    int4 affUp;
    int4 mchUp;
    
    int4 scrDown;
    int4 affDown;
    int4 mchDown;
    
    int2 wBus;
    int del;
    
    for (int j = 0; j < cols * 4; ++j) {
        hBus[j * width_ + tid] = make_int2(0, SCORE_MIN);
    }
    
    for (int i = 0; i < rowsPadded_; i += 8) {
    
        scrUp = INT4_ZERO;
        affUp = INT4_SCORE_MIN;
        mchUp = INT4_ZERO;
        
        scrDown = INT4_ZERO;
        affDown = INT4_SCORE_MIN;
        mchDown = INT4_ZERO;
        
        for (int j = 0; j < cols; ++j) {
        
            int columnCodes = tex2D(seqsTexture, colOff, j + rowOff);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
            
                wBus = hBus[(j * 4 + k) * width_ + tid];
                
                char code = (columnCodes >> (k << 3));
                char4 rowScores = tex2D(qpTexture, code, i / 4);
                
                del = max(wBus.x - gapOpen_, wBus.y - gapExtend_);
                affUp.x = max(scrUp.x - gapOpen_, affUp.x - gapExtend_);
                scrUp.x = mchUp.x + rowScores.x; 
                scrUp.x = max(scrUp.x, del);
                scrUp.x = max(scrUp.x, affUp.x);
                scrUp.x = max(scrUp.x, 0);
                mchUp.x = wBus.x;
                score = max(score, scrUp.x);
                
                del = max(scrUp.x - gapOpen_, del - gapExtend_);
                affUp.y = max(scrUp.y - gapOpen_, affUp.y - gapExtend_);
                scrUp.y = mchUp.y + rowScores.y; 
                scrUp.y = max(scrUp.y, del);
                scrUp.y = max(scrUp.y, affUp.y);
                scrUp.y = max(scrUp.y, 0);
                mchUp.y = scrUp.x;
                score = max(score, scrUp.y);
                
                del = max(scrUp.y - gapOpen_, del - gapExtend_);
                affUp.z = max(scrUp.z - gapOpen_, affUp.z - gapExtend_);
                scrUp.z = mchUp.z + rowScores.z; 
                scrUp.z = max(scrUp.z, del);
                scrUp.z = max(scrUp.z, affUp.z);
                scrUp.z = max(scrUp.z, 0);
                mchUp.z = scrUp.y;
                score = max(score, scrUp.z);
                
                del = max(scrUp.z - gapOpen_, del - gapExtend_);
                affUp.w = max(scrUp.w - gapOpen_, affUp.w - gapExtend_);
                scrUp.w = mchUp.w + rowScores.w; 
                scrUp.w = max(scrUp.w, del);
                scrUp.w = max(scrUp.w, affUp.w);
                scrUp.w = max(scrUp.w, 0);
                mchUp.w = scrUp.z;
                score = max(score, scrUp.w);

                rowScores = tex2D(qpTexture, code, i / 4 + 1);
                
                del = max(scrUp.w - gapOpen_, del - gapExtend_);
                affDown.x = max(scrDown.x - gapOpen_, affDown.x - gapExtend_);
                scrDown.x = mchDown.x + rowScores.x; 
                scrDown.x = max(scrDown.x, del);
                scrDown.x = max(scrDown.x, affDown.x);
                scrDown.x = max(scrDown.x, 0);
                mchDown.x = scrUp.w;
                score = max(score, scrDown.x);
                
                del = max(scrDown.x - gapOpen_, del - gapExtend_);
                affDown.y = max(scrDown.y - gapOpen_, affDown.y - gapExtend_);
                scrDown.y = mchDown.y + rowScores.y; 
                scrDown.y = max(scrDown.y, del);
                scrDown.y = max(scrDown.y, affDown.y);
                scrDown.y = max(scrDown.y, 0);
                mchDown.y = scrDown.x;
                score = max(score, scrDown.y);
                
                del = max(scrDown.y - gapOpen_, del - gapExtend_);
                affDown.z = max(scrDown.z - gapOpen_, affDown.z - gapExtend_);
                scrDown.z = mchDown.z + rowScores.z; 
                scrDown.z = max(scrDown.z, del);
                scrDown.z = max(scrDown.z, affDown.z);
                scrDown.z = max(scrDown.z, 0);
                mchDown.z = scrDown.y;
                score = max(score, scrDown.z);
                
                del = max(scrDown.z - gapOpen_, del - gapExtend_);
                affDown.w = max(scrDown.w - gapOpen_, affDown.w - gapExtend_);
                scrDown.w = mchDown.w + rowScores.w; 
                scrDown.w = max(scrDown.w, del);
                scrDown.w = max(scrDown.w, affDown.w);
                scrDown.w = max(scrDown.w, 0);
                mchDown.w = scrDown.z;
                score = max(score, scrDown.w);
                
                wBus.x = scrDown.w;
                wBus.y = del;
                
                hBus[(j * 4 + k) * width_ + tid] = wBus;
            }
        }
    }
    
    scores[id] = score;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// QUERY PROFILE

static QueryProfile* createQueryProfile(Chain* query, Scorer* scorer) {

    int rows = chainGetLength(query);
    int rowsGpu = rows + (8 - rows % 8) % 8;
    
    int width = scorerGetMaxCode(scorer) + 1;
    int height = rowsGpu / 4;

    char* row = (char*) malloc(rows * sizeof(char));
    chainCopyCodes(query, row);

    size_t size = width * height * sizeof(char4);
    char4* data = (char4*) malloc(size);
    memset(data, 0, size);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width - 1; ++j) {
            char4 scr;
            scr.x = i * 4 + 0 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 0], j);
            scr.y = i * 4 + 1 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 1], j);
            scr.z = i * 4 + 2 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 2], j);
            scr.w = i * 4 + 3 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 3], j);
            data[i * width + j] = scr;
        }
    }
    
    free(row);
    
    QueryProfile* queryProfile = (QueryProfile*) malloc(sizeof(QueryProfile));
    queryProfile->data = data;
    queryProfile->width = width;
    queryProfile->height = height;
    queryProfile->length = rows;
    queryProfile->size = size;
    
    return queryProfile;
}

static void deleteQueryProfile(QueryProfile* queryProfile) {
    free(queryProfile->data);
    free(queryProfile);
}

static QueryProfileGpu* createQueryProfileGpu(QueryProfile* queryProfile) {

    int width = queryProfile->width;
    int height = queryProfile->height;
    
    size_t size = queryProfile->size;
    char4* data = queryProfile->data;
    cudaArray* dataGpu;
    
    CUDA_SAFE_CALL(cudaMallocArray(&dataGpu, &qpTexture.channelDesc, width, height)); 
    CUDA_SAFE_CALL(cudaMemcpyToArray (dataGpu, 0, 0, data, size, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTextureToArray(qpTexture, dataGpu));
    qpTexture.addressMode[0] = cudaAddressModeClamp;
    qpTexture.addressMode[1] = cudaAddressModeClamp;
    qpTexture.filterMode = cudaFilterModePoint;
    qpTexture.normalized = false;
    
    size_t queryProfileGpuSize = sizeof(QueryProfileGpu);
    QueryProfileGpu* queryProfileGpu = (QueryProfileGpu*) malloc(queryProfileGpuSize);
    queryProfileGpu->data = dataGpu;
    
    return queryProfileGpu;
}

static void deleteQueryProfileGpu(QueryProfileGpu* queryProfileGpu) {
    CUDA_SAFE_CALL(cudaFreeArray(queryProfileGpu->data));
    CUDA_SAFE_CALL(cudaUnbindTexture(qpTexture));
    free(queryProfileGpu);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// UTILS

static int int2CmpY(const void* a_, const void* b_) {

    int2 a = *((int2*) a_);
    int2 b = *((int2*) b_);
    
    return a.y - b.y;
}

//------------------------------------------------------------------------------
//******************************************************************************

#endif // __CUDACC__

