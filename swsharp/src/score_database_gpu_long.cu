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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "cuda_utils.h"
#include "error.h"
#include "scorer.h"
#include "thread.h"
#include "utils.h"

#include "score_database_gpu_long.h"

#define THREADS   128
#define BLOCKS    240

#define INT2_ZERO make_int2(0, 0)
#define SCORE4_MIN make_int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

struct LongDatabase {
    int length;
};

typedef struct LongDatabaseGpu {

} LongDatabaseGpu;

typedef struct Context {
    int** scores; 
    int type;
    Chain** queries;
    int queriesLen;
    LongDatabase* longDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int* cards;
    int cardsLen;
} Context;

//******************************************************************************
// PUBLIC

extern LongDatabase* longDatabaseCreate(Chain** database, int databaseLen);

extern void longDatabaseDelete(LongDatabase* longDatabase);

extern void scoreLongDatabaseGpu(int** scores, int type, Chain* query, 
    LongDatabase* longDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread);

extern void scoreLongDatabasesGpu(int** scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void scoreDatabase(int** scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

static void* scoreDatabaseThread(void* param);

// cpu kernels
static void* kernel(void* param);

// gpu database preparation
static LongDatabaseGpu* longDatabaseGpuCreate(LongDatabase* longDatabase,
    int* indexes, int indexesLen);

static void longDatabaseGpuDelete(LongDatabaseGpu* longDatabaseGpu);

// gpu kernels



//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern LongDatabase* longDatabaseCreate(Chain** database, int databaseLen) {
    
    LongDatabase* longDatabase = 
        (LongDatabase*) malloc(sizeof(struct LongDatabase));
    
    longDatabase->length = databaseLen;
    
    return longDatabase;
}

extern void longDatabaseDelete(LongDatabase* longDatabase) {
    
    free(longDatabase);
    longDatabase = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

extern void scoreLongDatabaseGpu(int** scores, int type, Chain* query, 
    LongDatabase* longDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, &query, 1, longDatabase, scorer, indexes, 
        indexesLen, cards, cardsLen, thread);
}

extern void scoreLongDatabasesGpu(int** scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, queries, queriesLen, longDatabase, scorer,
        indexes, indexesLen, cards, cardsLen, thread);
}

//------------------------------------------------------------------------------

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// DATABASE SCORING

static void scoreDatabase(int** scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    
    ASSERT(cardsLen > 0, "no GPUs available");
    
    Context* param = (Context*) malloc(sizeof(Context));
    
    param->scores = scores;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->longDatabase = longDatabase;
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

static void* scoreDatabaseThread(void* param) {

    Context* context = (Context*) param;
    
    int** scores = context->scores;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    LongDatabase* longDatabase = context->longDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    //**************************************************************************
    // SOLVE MULTICARDED
    
    *scores = (int*) malloc(queriesLen * longDatabase->length * sizeof(int));
    
    for (int i = 0; i < longDatabase->length * queriesLen; i++) {
        (*scores)[i] = SCORE_MIN;
    }
    
    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(param);
    
    //**************************************************************************
    
    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU DATABASE PREPARE

static LongDatabaseGpu* longDatabaseGpuCreate(LongDatabase* longDatabase,
    int* indexes, int indexesLen) {
    
    LongDatabaseGpu* longDatabaseGpu = 
        (LongDatabaseGpu*) malloc(sizeof(struct LongDatabaseGpu));
    
    return longDatabaseGpu;
}

static void longDatabaseGpuDelete(LongDatabaseGpu* longDatabaseGpu) {

    free(longDatabaseGpu);
    longDatabaseGpu = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

//------------------------------------------------------------------------------

//******************************************************************************
