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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "cpu_module.h"
#include "cuda_utils.h"
#include "error.h"
#include "scorer.h"
#include "score_database_gpu_long.h"
#include "score_database_gpu_short.h"
#include "thread.h"
#include "threadpool.h"
#include "utils.h"

#include "gpu_module.h"

#define CPU_PACKET_LEN     200
#define MAX_CPU_LEN        20
#define MAX_SHORT_LEN      2800

typedef struct Context {
    int** scores;
    int type;
    Chain** queries; 
    int queriesLen;
    ChainDatabaseGpu* chainDatabaseGpu;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int* cards;
    int cardsLen;
} Context;

typedef struct CpuDatabase {
    Chain** database;
    int databaseLen;
    int* databaseIdx;
    int databaseIdxLen;
} CpuDatabase;

typedef struct CpuDatabaseContext {
    Chain* query;
    Chain* target;
    int* score;
} CpuDatabaseContext;

typedef struct CpuDatabaseContexts {
    CpuDatabaseContext* contexts;
    int contextsLen;
    int type;
    Scorer* scorer;
} CpuDatabaseContexts;

struct ChainDatabaseGpu {
    Chain** database;
    int databaseLen;
    int thresholded;
    int* order;
    int* position;
    CpuDatabase* cpuDatabase;
    ShortDatabase* shortDatabase;
    LongDatabase* longDatabase;
};

//******************************************************************************
// PUBLIC

extern ChainDatabaseGpu* chainDatabaseGpuCreate(Chain** database, int databaseLen,
    int* cards, int cardsLen);

extern void chainDatabaseGpuDelete(ChainDatabaseGpu* chainDatabaseGpu);

extern void scoreDatabaseGpu(int** scores, int type, Chain* query, 
    ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);
    
extern void scoreDatabasesGpu(int** scores, int type, Chain** queries, 
    int queriesLen, ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen, Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void scoreDatabase(int** scores, int type, Chain** queries, 
    int queriesLen, ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen, Thread* thread);
    
static void* scoreDatabaseThread(void* param);

static void filterIndexesArray(int** indexesNew, int* indexesNewLen, 
    int* indexes, int indexesLen, int minIndex, int maxIndex);

// cpu database
static CpuDatabase* cpuDatabaseCreate(Chain** database, int databaseLen,
    int minLen, int maxLen);

static void cpuDatabaseDelete();

static void* cpuDatabaseScore(void* param);

static void* cpuDatabaseScoreThread(void* param);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern ChainDatabaseGpu* chainDatabaseGpuCreate(Chain** database, int databaseLen,
    int* cards, int cardsLen) {

    if (cardsLen == 0) {
        return NULL;
    }

    // create databases
    CpuDatabase* cpuDatabase = cpuDatabaseCreate(database, databaseLen, 
        0, MAX_CPU_LEN);

    ShortDatabase* shortDatabase = shortDatabaseCreate(database, databaseLen, 
        MAX_CPU_LEN, MAX_SHORT_LEN, cards, cardsLen);
        
    LongDatabase* longDatabase = longDatabaseCreate(database, databaseLen, 
        MAX_SHORT_LEN, INT_MAX, cards, cardsLen);
    
    // save struct
    ChainDatabaseGpu* chainDatabaseGpu = 
        (ChainDatabaseGpu*) malloc(sizeof(struct ChainDatabaseGpu));
    
    chainDatabaseGpu->database = database;
    chainDatabaseGpu->databaseLen = databaseLen;
    chainDatabaseGpu->cpuDatabase = cpuDatabase;
    chainDatabaseGpu->shortDatabase = shortDatabase;
    chainDatabaseGpu->longDatabase = longDatabase;

    return chainDatabaseGpu;
}

extern void chainDatabaseGpuDelete(ChainDatabaseGpu* chainDatabaseGpu) {

    if (chainDatabaseGpu != 0) {

        cpuDatabaseDelete(chainDatabaseGpu->cpuDatabase);
        shortDatabaseDelete(chainDatabaseGpu->shortDatabase);
        longDatabaseDelete(chainDatabaseGpu->longDatabase);

        free(chainDatabaseGpu);
    }
}

extern void scoreDatabaseGpu(int** scores, int type, Chain* query, 
    ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, &query, 1, chainDatabaseGpu, scorer, indexes, 
        indexesLen, cards, cardsLen, thread);
}

extern void scoreDatabasesGpu(int** scores, int type, Chain** queries, 
    int queriesLen, ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, queries, queriesLen, chainDatabaseGpu, scorer,
        indexes, indexesLen, cards, cardsLen, thread);
}

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// ENTRY

static void scoreDatabase(int** scores, int type, Chain** queries, 
    int queriesLen, ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen, Thread* thread) {

    ASSERT(cardsLen > 0, "no GPUs available");
    
    Context* param = (Context*) malloc(sizeof(Context));

    param->scores = scores;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->chainDatabaseGpu = chainDatabaseGpu;
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
// SOLVE

static void* scoreDatabaseThread(void* param) {

    Context* context = (Context*) param;
    
    int** scores = context->scores;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    ChainDatabaseGpu* chainDatabaseGpu = context->chainDatabaseGpu;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    CpuDatabase* cpuDatabase = chainDatabaseGpu->cpuDatabase;
    ShortDatabase* shortDatabase = chainDatabaseGpu->shortDatabase;
    LongDatabase* longDatabase = chainDatabaseGpu->longDatabase;
    
    int databaseLen = chainDatabaseGpu->databaseLen;
    
    int i, j;
    
    //**************************************************************************
    // FILTER INDEXES
    
    int* indexesNew = NULL;
    int indexesNewLen;
    
    filterIndexesArray(&indexesNew, &indexesNewLen, indexes, indexesLen, 
        0, databaseLen);
    
    //**************************************************************************
        
    //**************************************************************************
    // INIT RESULTS
    
    *scores = (int*) malloc(queriesLen * databaseLen * sizeof(int));

    for (i = 0; i < queriesLen; ++i) {
        for (j = 0; j < databaseLen; ++j) {
            (*scores)[i * databaseLen + j] = NO_SCORE;
        }
    }

    //**************************************************************************
    
    //**************************************************************************
    // SOLVE MULTICARDED
    TIMER_START("Database solving GPU");

    Thread thread;
    if (cpuDatabase != NULL) {
        threadCreate(&thread, cpuDatabaseScore, context);
    }

    TIMER_START("Short solve");
    
    scoreShortDatabasesGpu(*scores, type, queries, queriesLen, 
        shortDatabase, scorer, indexesNew, indexesNewLen, cards, cardsLen, NULL);

    TIMER_STOP;
        
    TIMER_START("Long solve");
    
    scoreLongDatabasesGpu(*scores, type, queries, queriesLen, 
        longDatabase, scorer, indexesNew, indexesNewLen, cards, cardsLen, NULL);
        
    TIMER_STOP;

    if (cpuDatabase != NULL) {
        threadJoin(thread);
    }

    TIMER_STOP;

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY
    
    free(indexesNew);
    
    free(param);
    
    //**************************************************************************
    
    return NULL;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU DATABASE

static CpuDatabase* cpuDatabaseCreate(Chain** database, int databaseLen,
    int minLen, int maxLen) {

    int i;

    // calculate length
    int length = 0;

    for (i = 0; i < databaseLen; ++i) {
    
        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            length++;
        }
    }

    if (length == 0) {
        return NULL;
    }

    // create indexes array
    int* indexes = (int*) malloc(length * sizeof(int));
    int indexesLen = 0;

    for (i = 0; i < databaseLen; ++i) {
    
        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            indexes[indexesLen++] = i;
        }
    }

    CpuDatabase* cpuDatabase = 
        (CpuDatabase*) malloc(sizeof(struct CpuDatabase));

    cpuDatabase->database = database;
    cpuDatabase->databaseLen = databaseLen;
    cpuDatabase->databaseIdx = indexes;
    cpuDatabase->databaseIdxLen = indexesLen;

    return cpuDatabase;
}

static void cpuDatabaseDelete(CpuDatabase* cpuDatabase) {
    if (cpuDatabase != NULL) {
        free(cpuDatabase->databaseIdx);
        free(cpuDatabase);
    }
}

static void* cpuDatabaseScore(void* param) {

    TIMER_START("Cpu database solve");

    Context* context = (Context*) param;

    int* scores = *(context->scores);
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    CpuDatabase* cpuDatabase = context->chainDatabaseGpu->cpuDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;

    Chain** database = cpuDatabase->database;
    int databaseLen = cpuDatabase->databaseLen;
    int* databaseIdx = cpuDatabase->databaseIdx;
    int databaseIdxLen = cpuDatabase->databaseIdxLen;

    int i, j;

    //**************************************************************************
    // CREATE SOLVING MASK

    char* mask;
    if (indexes == NULL) {
        mask = NULL;
    } else {

        size_t maskSize = databaseLen * sizeof(char);
        mask = (char*) malloc(maskSize);
        memset(mask, 0, maskSize);

        for (i = 0; i < indexesLen; ++i) {
            mask[indexes[i]] = 1;
        }
    }

    //**************************************************************************

    //**************************************************************************
    // SOLVE MULTITHREADED

    int maxLen = databaseIdxLen * queriesLen;

    size_t contextsSize = maxLen * sizeof(CpuDatabaseContext);
    CpuDatabaseContext* contexts = (CpuDatabaseContext*) malloc(contextsSize);
    int contextIdx = 0;

    size_t packedSize = (maxLen / CPU_PACKET_LEN + 1) * sizeof(CpuDatabaseContexts);
    CpuDatabaseContexts* packed = (CpuDatabaseContexts*) malloc(packedSize);
    int packedIdx = 0;

    size_t tasksSize = (maxLen / CPU_PACKET_LEN + 1) * sizeof(ThreadPoolTask*);
    ThreadPoolTask** tasks = (ThreadPoolTask**) malloc(tasksSize);

    CpuDatabaseContext* start = contexts;
    int length = 0;

    for (i = 0; i < queriesLen; ++i) {

        Chain* query = queries[i];
        int lastQuery = i == queriesLen - 1;

        for (j = 0; j < databaseIdxLen; ++j) {

            int idx = databaseIdx[j];
            int last = lastQuery && j == databaseIdxLen - 1;

            if (mask != NULL && mask[idx] == 0) {
                scores[i * databaseLen + idx] = NO_SCORE;
            }

            Chain* target = database[idx];

            contexts[contextIdx].query = query;
            contexts[contextIdx].target = target;
            contexts[contextIdx].score = scores + i * databaseLen + idx;
            contextIdx++;

            length++;

            if (contextIdx % CPU_PACKET_LEN == 0 || last) {

                packed[packedIdx].contexts = start;
                packed[packedIdx].contextsLen = length;
                packed[packedIdx].type = type;
                packed[packedIdx].scorer = scorer;

                tasks[packedIdx] = threadPoolSubmit(cpuDatabaseScoreThread, 
                    &(packed[packedIdx]));

                packedIdx++;

                start = start + length;
                length = 0;
            }
        }
    }

    for (i = 0; i < packedIdx; ++i) {
        threadPoolTaskWait(tasks[i]);
        threadPoolTaskDelete(tasks[i]);
    }

    free(tasks);
    free(packed);
    free(contexts);

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY
    
    if (mask != NULL) {
        free(mask);
    }
    
    //**************************************************************************

    TIMER_STOP;

    return NULL;
}

static void* cpuDatabaseScoreThread(void* param) {

    CpuDatabaseContexts* context = (CpuDatabaseContexts*) param;

    CpuDatabaseContext* contexts = context->contexts;
    int contextsLen = context->contextsLen;
    int type = context->type;
    Scorer* scorer = context->scorer;

    int i; 
    for (i = 0; i < contextsLen; ++i) {

        Chain* query = contexts[i].query;
        Chain* target = contexts[i].target;
        int* score = contexts[i].score;

        *score = scorePairCpu(type, query, target, scorer);
    }

    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// UTILS

static void filterIndexesArray(int** indexesNew, int* indexesNewLen, 
    int* indexes, int indexesLen, int minIndex, int maxIndex) {
    
    if (indexes == NULL) {
        *indexesNew = NULL;
        *indexesNewLen = 0;
        return;
    }
    
    *indexesNew = (int*) malloc(indexesLen * sizeof(int));
    *indexesNewLen = 0;
    
    int i;
    for (i = 0; i < indexesLen; ++i) {
    
        int idx = indexes[i];
        
        if (idx >= minIndex && idx <= maxIndex) {
            (*indexesNew)[*indexesNewLen] = idx;
            (*indexesNewLen)++;
        }
    }
}

//------------------------------------------------------------------------------
//******************************************************************************
