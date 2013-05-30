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

#include "align.h"
#include "alignment.h"
#include "cpu_module.h"
#include "chain.h"
#include "constants.h"
#include "db_alignment.h"
#include "error.h"
#include "gpu_module.h"
#include "post_proc.h"
#include "scorer.h"
#include "thread.h"
#include "utils.h"

#include "database.h"

#define THREADS         4
#define GPU_MIN_CELLS   10000000.0

typedef struct Context {
    DbAlignment*** dbAlignments;
    int* dbAlignmentsLen;
    int type;
    Chain** queries;
    int queriesLen;
    ChainDatabase* chainDatabase;
    Scorer* scorer;
    int maxAlignments;
    ValueFunction valueFunction;
    void* valueFunctionParam;
    double valueThreshold;
    int* indexes;
    int indexesLen;
    int* cards;
    int cardsLen;
    Thread* thread;
} Context;

typedef struct DbAlignmentData {
    int idx;
    int score;
    double value;
} DbAlignmentData;

typedef struct AlignContext {
    DbAlignment*** dbAlignments;
    DbAlignmentData** dbAlignmentsData;
    int* dbAlignmentsLen;
    int type;
    Chain** queries;
    int queriesLen;
    Chain** database;
    int databaseStart;
    Scorer* scorer;
    int* cards;
    int cardsLen;
    int thread;
    int threads;
} AlignContext;

typedef struct ExtractContext {
    DbAlignmentData** dbAlignmentsData;
    int* dbAlignmentsLen;
    Chain** queries;
    int queriesLen;
    Chain** database;
    int databaseStart;
    int databaseLen;
    int* scores;
    int maxAlignments;
    ValueFunction valueFunction;
    void* valueFunctionParam;
    double valueThreshold;
    int thread;
    int threads;
} ExtractContext;

typedef struct IntDouble {
    int x;
    double y;
} IntDouble;

struct ChainDatabase {
    ChainDatabaseGpu* chainDatabaseGpu;
    Chain** database;
    int databaseStart;
    int databaseLen;
    long databaseElems;
};

//******************************************************************************
// PUBLIC

extern ChainDatabase* chainDatabaseCreate(Chain** database, int databaseStart, 
    int databaseLen);

extern void chainDatabaseDelete(ChainDatabase* chainDatabase);

extern void alignDatabase(DbAlignment*** dbAlignments, int* dbAlignmentsLen, 
    int type, Chain* query, ChainDatabase* chainDatabase, Scorer* scorer, 
    int maxAlignments, ValueFunction valueFunction, void* valueFunctionParam, 
    double valueThreshold, int* indexes, int indexesLen, int* cards, 
    int cardsLen, Thread* thread);
    
extern void shotgunDatabase(DbAlignment**** dbAlignments, int** dbAlignmentsLen, 
    int type, Chain** queries, int queriesLen, ChainDatabase* chainDatabase, 
    Scorer* scorer, int maxAlignments, ValueFunction valueFunction, 
    void* valueFunctionParam, double valueThreshold, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void databaseSearch(DbAlignment*** dbAlignments, int* dbAlignmentsLen, 
    int type, Chain** queries, int queriesLen, ChainDatabase* chainDatabase, 
    Scorer* scorer, int maxAlignments, ValueFunction valueFunction, 
    void* valueFunctionParam, double valueThreshold, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

static void* databaseSearchThread(void* param);

static void* alignThread(void* param);

static void* extractThread(void* param);

static void scoreDatabasesCpu(int** scores, int type, Chain** queries, 
    int queriesLen, Chain** database, int databaseLen, Scorer* scorer, 
    int* indexes, int indexesLen);

static void filterIndexesArray(int** indexesNew, int* indexesNewLen, 
    int* indexes, int indexesLen, int minIndex, int maxIndex);

static int intDoubleCmp(const void* a_, const void* b_);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern ChainDatabase* chainDatabaseCreate(Chain** database, int databaseStart, 
    int databaseLen) {
    
    ChainDatabase* db = (ChainDatabase*) malloc(sizeof(struct ChainDatabase));
    
    TIMER_START("Creating database");
    
    db->database = database;
    db->databaseStart = databaseStart;
    db->databaseLen = databaseLen;
    
    int i;
    long databaseElems = 0;
    for (i = 0; i < databaseLen; ++i) {
        databaseElems += chainGetLength(database[databaseStart + i]);
    }
    db->databaseElems = databaseElems;
    
    Chain** databaseGpu = database + databaseStart;
    db->chainDatabaseGpu = chainDatabaseGpuCreate(databaseGpu, databaseLen);
    
    TIMER_STOP;
    
    return db;
}

extern void chainDatabaseDelete(ChainDatabase* chainDatabase) {

    chainDatabaseGpuDelete(chainDatabase->chainDatabaseGpu);
    
    free(chainDatabase); 
    chainDatabase = NULL;
}

extern void alignDatabase(DbAlignment*** dbAlignments, int* dbAlignmentsLen, 
    int type, Chain* query, ChainDatabase* chainDatabase, Scorer* scorer, 
    int maxAlignments, ValueFunction valueFunction, void* valueFunctionParam, 
    double valueThreshold, int* indexes, int indexesLen, int* cards, 
    int cardsLen, Thread* thread) {

    databaseSearch(dbAlignments, dbAlignmentsLen, type, &query, 1,
        chainDatabase, scorer, maxAlignments, valueFunction, valueFunctionParam,
        valueThreshold, indexes, indexesLen, cards, cardsLen, thread);
}

extern void shotgunDatabase(DbAlignment**** dbAlignments, int** dbAlignmentsLen, 
    int type, Chain** queries, int queriesLen, ChainDatabase* chainDatabase, 
    Scorer* scorer, int maxAlignments, ValueFunction valueFunction, 
    void* valueFunctionParam, double valueThreshold, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {

    *dbAlignments = (DbAlignment***) malloc(queriesLen * sizeof(DbAlignment**));
    *dbAlignmentsLen = (int*) malloc(queriesLen * sizeof(int));
    
    databaseSearch(*dbAlignments, *dbAlignmentsLen, type, queries, queriesLen,
        chainDatabase, scorer, maxAlignments, valueFunction, valueFunctionParam, 
        valueThreshold, indexes, indexesLen, cards, cardsLen, thread);
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// SEARCH

static void databaseSearch(DbAlignment*** dbAlignments, int* dbAlignmentsLen, 
    int type, Chain** queries, int queriesLen, ChainDatabase* chainDatabase, 
    Scorer* scorer, int maxAlignments, ValueFunction valueFunction, 
    void* valueFunctionParam, double valueThreshold, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    
    Context* param = (Context*) malloc(sizeof(Context));
    
    param->dbAlignments = dbAlignments;
    param->dbAlignmentsLen = dbAlignmentsLen;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->chainDatabase = chainDatabase;
    param->scorer = scorer;
    param->maxAlignments = maxAlignments;
    param->valueFunction = valueFunction;
    param->valueFunctionParam = valueFunctionParam;
    param->valueThreshold = valueThreshold;
    param->indexes = indexes;
    param->indexesLen = indexesLen;
    param->cards = cards;
    param->cardsLen = cardsLen;
    
    if (thread == NULL) {
        databaseSearchThread(param);
    } else {
        threadCreate(thread, databaseSearchThread, (void*) param);
    }
}

static void* databaseSearchThread(void* param) {

    Context* context = (Context*) param;
    
    DbAlignment*** dbAlignments = context->dbAlignments;
    int* dbAlignmentsLen = context->dbAlignmentsLen;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    ChainDatabase* chainDatabase = context->chainDatabase;
    Scorer* scorer = context->scorer;
    int maxAlignments = context->maxAlignments;
    ValueFunction valueFunction = context->valueFunction;
    void* valueFunctionParam = context->valueFunctionParam;
    double valueThreshold = context->valueThreshold;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    Chain** database = chainDatabase->database;
    int databaseStart = chainDatabase->databaseStart;
    int databaseLen = chainDatabase->databaseLen;
    long databaseElems = chainDatabase->databaseElems;
    ChainDatabaseGpu* chainDatabaseGpu = chainDatabase->chainDatabaseGpu;
    
    TIMER_START("Database search");
    
    int i;
    
    //**************************************************************************
    // FIX INDEXES 
    
    int* indexes;
    int indexesLen;
    
    filterIndexesArray(&indexes, &indexesLen, context->indexes, 
        context->indexesLen, databaseStart, databaseStart + databaseLen - 1);

    for (i = 0; i < indexesLen; ++i) {
        indexes[i] -= databaseStart;
    }    
    
    //**************************************************************************
    
    //**************************************************************************
    // FIX ARGUMENTS 
    
    if (indexes != NULL) {
        maxAlignments = MIN(indexesLen, maxAlignments);
    }
    
    if (maxAlignments < 0) {
        maxAlignments = databaseLen;
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CALCULATE CELL NUMBER
    
    long queriesElems = 0;
    for (i = 0; i < queriesLen; ++i) {
        queriesElems += chainGetLength(queries[i]);
    }
    
    if (indexes != NULL) {
    
        databaseElems = 0;
        
        for (i = 0; i < indexesLen; ++i) {
            databaseElems += chainGetLength(database[databaseStart + indexes[i]]);
        }
    }
    
    double cells = (double) queriesElems * databaseElems;
    
    //**************************************************************************
    
    //**************************************************************************
    // CALCULATE SCORES
    
    int* scores;
    
    if (cells < GPU_MIN_CELLS || cardsLen == 0) {
        scoreDatabasesCpu(&scores, type, queries, queriesLen, 
            database + databaseStart, databaseLen, scorer, indexes, indexesLen);
    } else {
        scoreDatabasesGpu(&scores, type, queries, queriesLen, chainDatabaseGpu, 
            scorer, indexes, indexesLen, cards, cardsLen, NULL);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // EXTRACT BEST CHAINS AND SAVE THEIR DATA MULTITHREADED
    
    DbAlignmentData** dbAlignmentsData = 
        (DbAlignmentData**) malloc(queriesLen * sizeof(DbAlignmentData*));

    int extractThreadsNmr = MIN(queriesLen, THREADS);
    
    Thread* extractThreads = 
        (Thread*) malloc((extractThreadsNmr - 1) * sizeof(Thread));
    
    ExtractContext* extractContexts = 
        (ExtractContext*) malloc(extractThreadsNmr * sizeof(ExtractContext));
    
    for (i = 0; i < extractThreadsNmr; ++i) {
        extractContexts[i].dbAlignmentsData = dbAlignmentsData;
        extractContexts[i].dbAlignmentsLen = dbAlignmentsLen;
        extractContexts[i].queries = queries;
        extractContexts[i].queriesLen = queriesLen;
        extractContexts[i].database = database;
        extractContexts[i].databaseStart = databaseStart;
        extractContexts[i].databaseLen = databaseLen;
        extractContexts[i].scores = scores;
        extractContexts[i].maxAlignments = maxAlignments;
        extractContexts[i].valueFunction = valueFunction;
        extractContexts[i].valueFunctionParam = valueFunctionParam;
        extractContexts[i].valueThreshold = valueThreshold;
        extractContexts[i].thread = i;
        extractContexts[i].threads = extractThreadsNmr;
    }
    
    for (i = 0; i < extractThreadsNmr - 1; ++i) {
        threadCreate(&extractThreads[i], extractThread, &extractContexts[i]);
    }
    
    extractThread(&extractContexts[extractThreadsNmr - 1]);
    
    // wait for the threads
    for (i = 0; i < extractThreadsNmr - 1; ++i) {
        threadJoin(extractThreads[i]);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // ALIGN BEST TARGETS MULTITHREADED
    
    int alignThreadsNmr = THREADS;
    Thread* alignThreads = 
        (Thread*) malloc((alignThreadsNmr - 1) * sizeof(Thread));
    
    for (i = 0; i < queriesLen; ++i) {
        dbAlignments[i] = 
            (DbAlignment**) malloc(dbAlignmentsLen[i] * sizeof(DbAlignment*));
    }
    
    AlignContext* alignContexts = 
        (AlignContext*) malloc(alignThreadsNmr * sizeof(AlignContext));
  
    for (i = 0; i < alignThreadsNmr; ++i) {
        alignContexts[i].dbAlignments = dbAlignments;
        alignContexts[i].dbAlignmentsData = dbAlignmentsData;
        alignContexts[i].dbAlignmentsLen = dbAlignmentsLen;
        alignContexts[i].type = type;
        alignContexts[i].queries = queries;
        alignContexts[i].queriesLen = queriesLen;
        alignContexts[i].database = database;
        alignContexts[i].databaseStart = databaseStart;
        alignContexts[i].scorer = scorer;
        alignContexts[i].cards = cards;
        alignContexts[i].cardsLen = cardsLen;
        alignContexts[i].thread = i;
        alignContexts[i].threads = alignThreadsNmr;
    }
    
    for (i = 0; i < alignThreadsNmr - 1; ++i) {
        threadCreate(&alignThreads[i], alignThread, &alignContexts[i]);
    }
    
    alignThread(&alignContexts[alignThreadsNmr - 1]);
    
    // wait for the threads
    for (i = 0; i < alignThreadsNmr - 1; ++i) {
        threadJoin(alignThreads[i]);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    for (i = 0; i < queriesLen; ++i) {
        free(dbAlignmentsData[i]);
    }
    free(dbAlignmentsData);
    
    free(extractThreads);
    free(extractContexts);
    free(alignThreads);
    free(alignContexts);
    
    free(indexes); // copy
    free(scores);

    free(param);
    
    //**************************************************************************
    
    TIMER_STOP;
        
    return NULL;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// THREADS

static void* alignThread(void* param) {

    AlignContext* context = (AlignContext*) param;
    
    DbAlignment*** dbAlignments = context->dbAlignments;
    DbAlignmentData** dbAlignmentsData = context->dbAlignmentsData;
    int* dbAlignmentsLen = context->dbAlignmentsLen;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain** database = context->database;
    int databaseStart = context->databaseStart;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    int thread = context->thread;
    int threads = context->threads;
    
    // distribute cards
    cards = cards + thread;
    cardsLen = thread < cardsLen;

    int i;
    for (i = 0; i < queriesLen; ++i) {
    
        Chain* query = queries[i];
        
        int j;
        for (j = thread; j < dbAlignmentsLen[i]; j+= threads) {
        
            DbAlignmentData* data = &(dbAlignmentsData[i][j]);
            
            int targetIdx = databaseStart + data->idx;
            Chain* target = database[targetIdx];

            // align
            Alignment* alignment;
            alignPair(&alignment, type, query, target, scorer, cards, 
                cardsLen, NULL);
            
            // check scores
            int s1 = alignmentGetScore(alignment);
            int s2 = data->score;
            
            ASSERT(s1 == s2, "Scores don't match %d %d", s1, s2);
            
            // extract info
            int queryStart = alignmentGetQueryStart(alignment);
            int queryEnd = alignmentGetQueryEnd(alignment);
            int targetStart = alignmentGetTargetStart(alignment); 
            int targetEnd = alignmentGetTargetEnd(alignment);
            Scorer* scorer = alignmentGetScorer(alignment);
            int pathLen = alignmentGetPathLen(alignment);
            
            char* path = (char*) malloc(pathLen);
            alignmentCopyPath(alignment, path);
            
            alignmentDelete(alignment);
            
            // create db alignment
            DbAlignment* dbAlignment = dbAlignmentCreate(query, queryStart, 
                queryEnd, i, target, targetStart, targetEnd, targetIdx, 
                data->value, data->score, scorer, path, pathLen);
    
            dbAlignments[i][j] = dbAlignment;
        }
    }

    return NULL;
}

static void* extractThread(void* param) {

    ExtractContext* context = (ExtractContext*) param;
    
    DbAlignmentData** dbAlignmentsData = context->dbAlignmentsData;
    int* dbAlignmentsLen = context->dbAlignmentsLen;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain** database = context->database;
    int databaseStart = context->databaseStart;
    int databaseLen = context->databaseLen;
    int* scores = context->scores;
    int maxAlignments = context->maxAlignments;
    ValueFunction valueFunction = context->valueFunction;
    void* valueFunctionParam = context->valueFunctionParam;
    double valueThreshold = context->valueThreshold;
    int thread = context->thread;
    int threads = context->threads;
    
    
    IntDouble* packed = (IntDouble*) malloc(databaseLen * sizeof(IntDouble));
    double* vals = (double*) malloc(databaseLen * sizeof(double));
    
    int i, j;
    
    for (i = thread; i < queriesLen; i += threads) {
    
        Chain* query = queries[i];
        int* queryScores = scores + i * databaseLen;
        
        valueFunction(vals, queryScores, query, database + databaseStart, 
            databaseLen, valueFunctionParam);

        int thresholded = 0;
    
        for (j = 0; j < databaseLen; ++j) {
        
            packed[j].x = j;
            packed[j].y = vals[j];
            
            if (packed[j].y <= valueThreshold) {
                thresholded++;
            }
        }
        
        int k = MIN(thresholded, maxAlignments);
        
        qselect((void*) packed, databaseLen, sizeof(IntDouble), k, intDoubleCmp);
        qsort((void*) packed, k, sizeof(IntDouble), intDoubleCmp);
        
        size_t dbAlignmentsDataSize = k * sizeof(DbAlignmentData);
        dbAlignmentsData[i] = (DbAlignmentData*) malloc(dbAlignmentsDataSize);
        
        dbAlignmentsLen[i] = k;
        
        for (j = 0; j < k; ++j) {
            dbAlignmentsData[i][j].idx = packed[j].x;
            dbAlignmentsData[i][j].value = packed[j].y;
            dbAlignmentsData[i][j].score = queryScores[packed[j].x];
        }
    }
    
    free(packed);
    free(vals);
    
    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU MODULES

static void scoreDatabasesCpu(int** scores, int type, Chain** queries, 
    int queriesLen, Chain** database, int databaseLen, Scorer* scorer, 
    int* indexes, int indexesLen) {
    
    TIMER_START("CPU database scoring");
    
    *scores = (int*) malloc(queriesLen * databaseLen * sizeof(int));
    
    int i, j;
    
    if (indexes == NULL) {
    
        for (i = 0; i < queriesLen; ++i) {
        
            Chain* query = queries[i];
            
            for (j = 0; j < databaseLen; ++j) {
            
                Chain* target = database[j];
                int score = scorePairCpu(type, query, target, scorer);
                
                (*scores)[i * databaseLen + j] = score;
            }
        }        
        
    } else {
    
        for (i = 0; i < queriesLen; ++i) {
            for (j = 0; j < databaseLen; ++j) {
                (*scores)[i * databaseLen + j] = NO_SCORE;
            }
        }
        
        for (i = 0; i < queriesLen; ++i) {
        
            Chain* query = queries[i];
            
            for (j = 0; j < indexesLen; ++j) {
            
                int idx = indexes[j];
                
                ASSERT(idx < databaseLen, "wrong index: %d", idx);

                Chain* target = database[idx];
                int score = scorePairCpu(type, query, target, scorer);
                
                (*scores)[i * databaseLen + idx] = score;
            }
        }    
    }
    
    TIMER_STOP;
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

static int intDoubleCmp(const void* a_, const void* b_) {

    IntDouble a = *((IntDouble*) a_);
    IntDouble b = *((IntDouble*) b_);
    
    if (a.y < b.y) return -1;
    if (a.y > b.y) return 1;
    return 0;
}

//------------------------------------------------------------------------------
//******************************************************************************
