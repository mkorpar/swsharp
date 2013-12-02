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

#define CPU_THREAD_CHUNK    32

#define MAX_CPU_LEN         20
#define MAX_SHORT_LEN       2800

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
    int databaseLen;
    Chain** filtered;
    int filteredLen;
    int* positions;
} CpuDatabase;

typedef struct CpuDatabaseContext {
    int* scores;
    int type;
    Chain* query;
    Chain** database;
    int databaseLen;
    Scorer* scorer;
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

    Chain** filtered = (Chain**) malloc(length * sizeof(Chain*));
    int filteredLen = 0;

    int* positions = (int*) malloc(length * sizeof(int));

    for (i = 0; i < databaseLen; ++i) {
    
        const int n = chainGetLength(database[i]);
        
        if (n >= minLen && n < maxLen) {
            filtered[filteredLen] = database[i];
            positions[filteredLen] = i;
            filteredLen++;
        }
    }

    CpuDatabase* cpuDatabase = 
        (CpuDatabase*) malloc(sizeof(struct CpuDatabase));

    cpuDatabase->databaseLen = databaseLen;
    cpuDatabase->filtered = filtered;
    cpuDatabase->filteredLen = filteredLen;
    cpuDatabase->positions = positions;

    return cpuDatabase;
}

static void cpuDatabaseDelete(CpuDatabase* cpuDatabase) {
    if (cpuDatabase != NULL) {
        free(cpuDatabase->filtered);
        free(cpuDatabase->positions);
        free(cpuDatabase);
    }
}

static void* cpuDatabaseScore(void* param) {

    Context* context = (Context*) param;

    int* scores_ = *(context->scores);
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    CpuDatabase* cpuDatabase = context->chainDatabaseGpu->cpuDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;

    int databaseLen = cpuDatabase->databaseLen;
    Chain** filtered_ = cpuDatabase->filtered;
    int filteredLen_ = cpuDatabase->filteredLen;
    int* positions_ = cpuDatabase->positions;

    Chain** filtered;
    int filteredLen;

    int* positions;

    int* scores;

    int i, j;

    if (cpuDatabase == NULL) {
        return NULL;
    }

    TIMER_START("Cpu database solve");

    //**************************************************************************
    // INIT STRUCTURES

    if (indexes == NULL) {

        filtered = filtered_;
        filteredLen = filteredLen_;

        positions = positions_;

    } else {

        char* mask = (char*) malloc(databaseLen * sizeof(char));
        memset(mask, 0, databaseLen * sizeof(char));

        for (i = 0; i < indexesLen; ++i) {
            mask[indexes[i]] = 1;
        }

        filtered = (Chain**) malloc(indexesLen * sizeof(Chain*));
        filteredLen = 0;

        positions = (int*) malloc(indexesLen * sizeof(int));

        for (i = 0; i < filteredLen_; ++i) {

            int idx = positions_[i];

            if (mask[idx]) {
                filtered[filteredLen] = filtered_[i];
                positions[filteredLen] = idx;
                filteredLen++;
            }
        }

        free(mask);
    }

    scores = (int*) malloc(filteredLen * queriesLen * sizeof(int));

    //**************************************************************************

    //**************************************************************************
    // SOLVE MULTITHREADED

    LOG("Cpu length %dx%d", queriesLen, filteredLen);

    int maxLen = (queriesLen * filteredLen) / CPU_THREAD_CHUNK + queriesLen; 
    int length = 0;

    size_t contextsSize = maxLen * sizeof(CpuDatabaseContext);
    CpuDatabaseContext* contexts = (CpuDatabaseContext*) malloc(contextsSize);

    size_t tasksSize = maxLen * sizeof(ThreadPoolTask*);
    ThreadPoolTask** tasks = (ThreadPoolTask**) malloc(tasksSize);

    for (i = 0; i < queriesLen; ++i) {
        for (j = 0; j < filteredLen; j += CPU_THREAD_CHUNK) {

            contexts[length].scores = scores + i * filteredLen + j;
            contexts[length].type = type;
            contexts[length].query = queries[i];
            contexts[length].database = filtered + j;
            contexts[length].databaseLen = MIN(CPU_THREAD_CHUNK, filteredLen - j);
            contexts[length].scorer = scorer;

            tasks[length] = threadPoolSubmit(cpuDatabaseScoreThread, &(contexts[length]));

            length++;
        }
    }

    for (i = 0; i < length; ++i) {
        threadPoolTaskWait(tasks[i]);
        threadPoolTaskDelete(tasks[i]);
    }

    free(tasks);
    free(contexts);

    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    if (indexes != NULL) {

        // init results, not all are solved
        for (i = 0; i < queriesLen; ++i) {
            for (j = 0; j < filteredLen_; ++j) {
                scores_[i * databaseLen + j] = NO_SCORE;
            }
        }
    }

    for (i = 0; i < queriesLen; ++i) {
        for (j = 0; j < filteredLen; ++j) {

            int idx = positions[j];
            int score = scores[i * filteredLen + j];

            scores_[i * databaseLen + idx] = score;
        }
    }

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    if (indexes != NULL) {
        free(filtered);
        free(positions);
    }

    free(scores);

    //**************************************************************************

    TIMER_STOP;

    return NULL;
}

static void* cpuDatabaseScoreThread(void* param) {

    CpuDatabaseContext* context = (CpuDatabaseContext*) param;

    int* scores = context->scores;
    int type = context->type;
    Chain* query = context->query;
    Chain** database = context->database;
    int databaseLen = context->databaseLen;
    Scorer* scorer = context->scorer;

    scoreDatabaseCpu(scores, type, query, database, databaseLen, scorer);

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
