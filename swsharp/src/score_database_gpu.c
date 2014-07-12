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

struct ChainDatabaseGpu {
    Chain** database;
    int databaseLen;
    int thresholded;
    int* order;
    int* position;
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

//******************************************************************************

//******************************************************************************
// PUBLIC

extern ChainDatabaseGpu* chainDatabaseGpuCreate(Chain** database, int databaseLen,
    int* cards, int cardsLen) {

    if (cardsLen == 0 || databaseLen == 0) {
        return NULL;
    }

    ShortDatabase* shortDatabase = shortDatabaseCreate(database, databaseLen, 
        0, MAX_SHORT_LEN, cards, cardsLen);
        
    LongDatabase* longDatabase = longDatabaseCreate(database, databaseLen, 
        MAX_SHORT_LEN, INT_MAX, cards, cardsLen);
    
    // save struct
    ChainDatabaseGpu* chainDatabaseGpu = 
        (ChainDatabaseGpu*) malloc(sizeof(struct ChainDatabaseGpu));
    
    chainDatabaseGpu->database = database;
    chainDatabaseGpu->databaseLen = databaseLen;
    chainDatabaseGpu->shortDatabase = shortDatabase;
    chainDatabaseGpu->longDatabase = longDatabase;

    return chainDatabaseGpu;
}

extern void chainDatabaseGpuDelete(ChainDatabaseGpu* chainDatabaseGpu) {

    if (chainDatabaseGpu != 0) {

        shortDatabaseDelete(chainDatabaseGpu->shortDatabase);
        longDatabaseDelete(chainDatabaseGpu->longDatabase);

        free(chainDatabaseGpu);
    }
}

extern size_t chainDatabaseGpuMemoryConsumption(Chain** database, int databaseLen) {

    size_t mem1 = shortDatabaseGpuMemoryConsumption(database, databaseLen,
        0, MAX_SHORT_LEN);
    size_t mem2 = longDatabaseGpuMemoryConsumption(database, databaseLen,
        MAX_SHORT_LEN, INT_MAX);

    return mem1 + mem2;
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

    TIMER_START("Short solve");
    
    scoreShortDatabasesGpu(*scores, type, queries, queriesLen, 
        shortDatabase, scorer, indexesNew, indexesNewLen, cards, cardsLen, NULL);

    TIMER_STOP;
        
    TIMER_START("Long solve");
    
    scoreLongDatabasesGpu(*scores, type, queries, queriesLen, 
        longDatabase, scorer, indexesNew, indexesNewLen, cards, cardsLen, NULL);
        
    TIMER_STOP;

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
