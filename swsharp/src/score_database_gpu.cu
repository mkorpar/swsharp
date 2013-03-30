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
#include "scorer.h"
#include "error.h"
#include "cuda_utils.h"
#include "utils.h"
#include "score_database_gpu_long.h"
#include "score_database_gpu_short.h"
#include "thread.h"

#include "gpu_module.h"

#define THRESHOLD       3000

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

typedef struct KernelContext {
    int* scoresShort;
    int* scoresLong;
    int type;
    Chain** queries; 
    int queriesLen;
    ChainDatabaseGpu* chainDatabaseGpu;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int card;
    int cardsLen;
} KernelContext;

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

extern ChainDatabaseGpu* chainDatabaseGpuCreate(Chain** database, int databaseLen);

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

//******************************************************************************

//******************************************************************************
// PUBLIC

extern ChainDatabaseGpu* chainDatabaseGpuCreate(Chain** database, int databaseLen) {

    // count thresholded
    int thresholded = 0;
    int total = 0;
    
    for (int i = 0; i < databaseLen; ++i) {
    
        int n = chainGetLength(database[i]);
        total += n;
        
        if (n < THRESHOLD) {
            thresholded++;
        }
    }
    
    int shortChainsLen = thresholded;
    int longChainsLen = databaseLen - thresholded;
    
    LOG("short: %d, long: %d", shortChainsLen, longChainsLen);
    LOG("total: %d", total);
    
    // seperate chains
    Chain** shortChains = (Chain**) malloc(shortChainsLen * sizeof(Chain*));
    Chain** longChains = (Chain**) malloc(longChainsLen * sizeof(Chain*));
    
    int* order = (int*) malloc(databaseLen * sizeof(int));
    
    for (int i = 0, s = 0, l = 0; i < databaseLen; ++i) {
        if (chainGetLength(database[i]) < THRESHOLD) {
            shortChains[s] = database[i];
            order[i] = s;
            s++;
        } else {
            longChains[l] = database[i];
            order[i] = thresholded + l;
            l++;
        }
    }
    
    int* position = (int*) malloc(databaseLen * sizeof(int));
    
    for (int i = 0; i < databaseLen; ++i) {
        position[order[i]] = i;
    }
    
    // create databases
    ShortDatabase* shortDatabase = shortDatabaseCreate(shortChains, shortChainsLen);
    LongDatabase* longDatabase = longDatabaseCreate(longChains, longChainsLen);
    
    free(shortChains);
    free(longChains);

    // save struct
    ChainDatabaseGpu* chainDatabaseGpu = 
        (ChainDatabaseGpu*) malloc(sizeof(struct ChainDatabaseGpu));
    
    chainDatabaseGpu->database = database;
    chainDatabaseGpu->databaseLen = databaseLen;
    chainDatabaseGpu->thresholded = thresholded;
    chainDatabaseGpu->order = order;
    chainDatabaseGpu->position = position;
    chainDatabaseGpu->longDatabase = longDatabase;
    chainDatabaseGpu->shortDatabase = shortDatabase;
    
    return chainDatabaseGpu;
}

extern void chainDatabaseGpuDelete(ChainDatabaseGpu* chainDatabaseGpu) {

    longDatabaseDelete(chainDatabaseGpu->longDatabase);
    shortDatabaseDelete(chainDatabaseGpu->shortDatabase);
    free(chainDatabaseGpu->order);
    free(chainDatabaseGpu->position);

    free(chainDatabaseGpu);
    chainDatabaseGpu = NULL;
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
    
    int databaseLen = chainDatabaseGpu->databaseLen;
    int thresholded = chainDatabaseGpu->thresholded;
    int* order = chainDatabaseGpu->order;
    int* position = chainDatabaseGpu->position;
    
    //**************************************************************************
    // CREATE REPLACED INDEXES
    
    int* shortIndexes = NULL;
    int shortIndexesLen = 0;
    
    int* longIndexes = NULL;
    int longIndexesLen = 0;
    
    // separete indexes if needed
    if (indexes != NULL) {
    
        shortIndexes = (int*) malloc(indexesLen * sizeof(int));
        longIndexes = (int*) malloc(indexesLen * sizeof(int));
        
        for (int i = 0; i < indexesLen; ++i) {
            
            int idx = indexes[i];
            ASSERT(idx < databaseLen, "wrong index: %d", idx);
            
            int ord = order[idx];
            
            if (ord < thresholded) {
                shortIndexes[shortIndexesLen++] = ord;
            } else {
                longIndexes[longIndexesLen++] = ord - thresholded;
            }
        }
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // SOLVE MULTICARDED
    
    TIMER_START("Database solving GPU");
    
    int* scoresShort;
    scoreShortDatabasesGpu(&scoresShort, type, queries, queriesLen, 
        chainDatabaseGpu->shortDatabase, scorer, shortIndexes, shortIndexesLen, 
        cards, cardsLen, NULL);
    
    int* scoresLong;
    scoreLongDatabasesGpu(&scoresLong, type, queries, queriesLen, 
        chainDatabaseGpu->longDatabase, scorer, longIndexes, longIndexesLen, 
        cards, cardsLen, NULL);
    

    TIMER_STOP;
    
    //**************************************************************************
    
    //**************************************************************************
    // SAVE RESULTS
    
    *scores = (int*) malloc(queriesLen * databaseLen * sizeof(int));

    for (int i = 0; i < queriesLen; ++i) {
    
        for (int j = 0; j < thresholded; ++j) {
            int scr = scoresShort[i * thresholded + j];
            (*scores)[i * databaseLen + position[j]] = scr;
        }
        
        for (int j = thresholded; j < databaseLen; ++j) {
            int scr = scoresLong[i * (databaseLen - thresholded) + j - thresholded];
            (*scores)[i * databaseLen + position[j]] = scr;
        }
    }
    
    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY
    
    free(shortIndexes);
    free(longIndexes);
    free(scoresLong);
    free(scoresShort);

    free(param);
    
    //**************************************************************************
    
    return NULL;
}

//******************************************************************************
