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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "align.h"
#include "alignment.h"
#include "cpu_module.h"
#include "chain.h"
#include "constants.h"
#include "cuda_utils.h"
#include "db_alignment.h"
#include "error.h"
#include "gpu_module.h"
#include "post_proc.h"
#include "scorer.h"
#include "thread.h"
#include "threadpool.h"
#include "utils.h"

#include "database.h"

#define THREADS             4
#define GPU_DB_MIN_CELLS    49000000ll
#define GPU_MIN_CELLS       1000000ll
#define GPU_MIN_LEN         256

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

typedef struct ExtractContext {
    DbAlignmentData** dbAlignmentData;
    int* dbAlignmentLen;
    Chain* query;
    Chain** database;
    int databaseLen;
    int* scores;
    int maxAlignments;
    ValueFunction valueFunction;
    void* valueFunctionParam;
    double valueThreshold;
} ExtractContext;

typedef struct AlignContext {
    DbAlignment** dbAlignment;
    int type;
    Chain* query;
    int queryIdx;
    Chain* target;
    int targetIdx;
    double value;
    int score;
    Scorer* scorer;
    int* cards;
    int cardsLen;
} AlignContext;

typedef struct AlignContexts {
    AlignContext* contexts;
    int contextsLen;
} AlignContexts;

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
    int databaseLen, int* cards, int cardsLen);

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

static void databaseSearchStep(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLen, int type, Chain** queries, int queriesStart, 
    int queriesLen, ChainDatabase* chainDatabase, Scorer* scorer, 
    int maxAlignments, ValueFunction valueFunction, void* valueFunctionParam, 
    double valueThreshold, int* indexes, int indexesLen, int* cards, 
    int cardsLen);

static void* alignThread(void* param);

static void* alignsThread(void* param);

static void* extractThread(void* param);

static void scoreDatabasesCpu(int** scores, int type, Chain** queries, 
    int queriesLen, Chain** database, int databaseLen, Scorer* scorer, 
    int* indexes, int indexesLen);

static void filterIndexesArray(int** indexesNew, int* indexesNewLen, 
    int* indexes, int indexesLen, int minIndex, int maxIndex);

static int dbAlignmentDataCmp(const void* a_, const void* b_);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern ChainDatabase* chainDatabaseCreate(Chain** database, int databaseStart, 
    int databaseLen, int* cards, int cardsLen) {
    
    ChainDatabase* db = (ChainDatabase*) malloc(sizeof(struct ChainDatabase));
    
    TIMER_START("Creating database");
    
    db->database = database + databaseStart;
    db->databaseStart = databaseStart;
    db->databaseLen = databaseLen;
    
    int i;
    long databaseElems = 0;
    for (i = 0; i < databaseLen; ++i) {
        databaseElems += chainGetLength(db->database[i]);
    }
    db->databaseElems = databaseElems;
    
    db->chainDatabaseGpu = chainDatabaseGpuCreate(db->database, databaseLen, 
        cards, cardsLen);
    
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
    
    int databaseStart = chainDatabase->databaseStart;
    int databaseLen = chainDatabase->databaseLen;
    
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
    // DO THE ALIGN
     
    double memory = (double) databaseLen * queriesLen * sizeof(int) + // scores
                    (double) MAX(0, maxAlignments) * queriesLen * 1024; // aligns
    memory = (memory * 1.1) / 1024.0 / 1024.0; // 10% offset and to MB
    
    // chop in pieces
    int steps = (int) ceil(memory / (1 * 1024.0));
    int queriesStep = queriesLen / steps;
    
    LOG("need %.2lfMB total, solving in %d steps", memory, steps);
    
    for (i = 0; i < steps; ++i) {
    
        int offset = i * queriesStep;
        int length = i == steps - 1 ? queriesLen - offset : queriesStep;
        
        databaseSearchStep(dbAlignments + offset, dbAlignmentsLen + offset, 
            type, queries + offset, offset, length, chainDatabase, scorer, 
            maxAlignments, valueFunction, valueFunctionParam, valueThreshold, 
            indexes, indexesLen, cards, cardsLen);
    }

    //**************************************************************************
 
    //**************************************************************************
    // CLEAN MEMORY

    free(indexes); // copy
    
    free(param);

    //**************************************************************************
    
    TIMER_STOP;
        
    return NULL;
}

static void databaseSearchStep(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLen, int type, Chain** queries, int queriesStart, 
    int queriesLen, ChainDatabase* chainDatabase, Scorer* scorer, 
    int maxAlignments, ValueFunction valueFunction, void* valueFunctionParam, 
    double valueThreshold, int* indexes, int indexesLen, int* cards, 
    int cardsLen) {
    
    Chain** database = chainDatabase->database;
    int databaseStart = chainDatabase->databaseStart;
    int databaseLen = chainDatabase->databaseLen;
    long databaseElems = chainDatabase->databaseElems;
    ChainDatabaseGpu* chainDatabaseGpu = chainDatabase->chainDatabaseGpu;
    
    int i, j, k;
    
    //**************************************************************************
    // CALCULATE CELL NUMBER
    
    long queriesElems = 0;
    for (i = 0; i < queriesLen; ++i) {
        queriesElems += chainGetLength(queries[i]);
    }
    
    if (indexes != NULL) {
    
        databaseElems = 0;
        
        for (i = 0; i < indexesLen; ++i) {
            databaseElems += chainGetLength(database[indexes[i]]);
        }
    }
    
    long long cells = (long long) queriesElems * databaseElems;
    
    //**************************************************************************
    
    //**************************************************************************
    // CALCULATE SCORES
    
    int* scores;
    
    if (cells < GPU_DB_MIN_CELLS || cardsLen == 0) {
        scoreDatabasesCpu(&scores, type, queries, queriesLen, database, 
            databaseLen, scorer, indexes, indexesLen);
    } else {
        scoreDatabasesGpu(&scores, type, queries, queriesLen, chainDatabaseGpu, 
            scorer, indexes, indexesLen, cards, cardsLen, NULL);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // EXTRACT BEST CHAINS AND SAVE THEIR DATA MULTITHREADED
    
    DbAlignmentData** dbAlignmentsData = 
        (DbAlignmentData**) malloc(queriesLen * sizeof(DbAlignmentData*));

    ExtractContext* eContexts = 
        (ExtractContext*) malloc(queriesLen * sizeof(ExtractContext));
    
    for (i = 0; i < queriesLen; ++i) {
        eContexts[i].dbAlignmentData = &(dbAlignmentsData[i]);
        eContexts[i].dbAlignmentLen = &(dbAlignmentsLen[i]);
        eContexts[i].query = queries[i];
        eContexts[i].database = database;
        eContexts[i].databaseLen = databaseLen;
        eContexts[i].scores = scores + i * databaseLen;
        eContexts[i].maxAlignments = maxAlignments;
        eContexts[i].valueFunction = valueFunction;
        eContexts[i].valueFunctionParam = valueFunctionParam;
        eContexts[i].valueThreshold = valueThreshold;
    }
    
    ThreadPoolTask** eTasks = 
        (ThreadPoolTask**) malloc(queriesLen * sizeof(ThreadPoolTask*));

    for (i = 0; i < queriesLen; ++i) {
        eTasks[i] = threadPoolSubmit(extractThread, (void*) &(eContexts[i]));
    }
    
    for (i = 0; i < queriesLen; ++i) {
        threadPoolTaskWait(eTasks[i]);
        threadPoolTaskDelete(eTasks[i]);
    }

    free(eContexts);
    free(eTasks);
    free(scores); // this is big, release immediately
    
    //**************************************************************************
    
    //**************************************************************************
    // ALIGN BEST TARGETS MULTITHREADED
    
    TIMER_START("Database aligning");
    
    // create structure
    for (i = 0; i < queriesLen; ++i) {
        size_t dbAlignmentsSize = dbAlignmentsLen[i] * sizeof(DbAlignment*);
        dbAlignments[i] = (DbAlignment**) malloc(dbAlignmentsSize);
    }
    
    // count tasks
    int aTasksLen = 0;
    for (i = 0; i < queriesLen; ++i) {
        aTasksLen += dbAlignmentsLen[i];
    }
    
    size_t aTasksSize = aTasksLen * sizeof(ThreadPoolTask*);
    ThreadPoolTask** aTasks = (ThreadPoolTask**) malloc(aTasksSize);

    size_t aContextsSize = aTasksLen * sizeof(AlignContext);
    AlignContext* aContextsCpu = (AlignContext*) malloc(aContextsSize);
    AlignContext* aContextsGpu = (AlignContext*) malloc(aContextsSize);
    int aContextsCpuLen = 0;
    int aContextsGpuLen = 0;
    
    for (i = 0, k = 0; i < queriesLen; ++i, ++k) {
    
        Chain* query = queries[i];
        int rows = chainGetLength(query);

        for (j = 0; j < dbAlignmentsLen[i]; ++j, ++k) {
            
            DbAlignmentData data = dbAlignmentsData[i][j];
            Chain* target = database[data.idx];

            int cols = chainGetLength(target);
            double cells = (double) rows * cols;

            AlignContext* context;
            if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
                context = &(aContextsCpu[aContextsCpuLen++]);
                context->cards = NULL;
                context->cardsLen = 0;
            } else {
                context = &(aContextsGpu[aContextsGpuLen++]);
            }
            
            context->dbAlignment = &(dbAlignments[i][j]);
            context->type = type;
            context->query = query;
            context->queryIdx = i;
            context->target = target;
            context->targetIdx = data.idx + databaseStart;
            context->value = data.value;
            context->score = data.score;
            context->scorer = scorer;
        }
    }
    
    LOG("Aligning %d cpu, %d gpu", aContextsCpuLen, aContextsGpuLen);

    for (i = 0; i < aContextsCpuLen; ++i) {
        aTasks[i] = threadPoolSubmit(alignThread, (void*) &(aContextsCpu[i]));
    }
    
    if (aContextsGpuLen) {

        int chunks = MIN(aContextsGpuLen, cardsLen);
        int cardsChunk = cardsLen / chunks;
        int cardsAdd = cardsLen % chunks;
        int contextChunk = aContextsGpuLen / chunks;
        int contextAdd = aContextsGpuLen % chunks;
        
        size_t contextsSize = chunks * sizeof(AlignContexts);
        AlignContexts* contexts = (AlignContexts*) malloc(contextsSize);

        int contextOff = 0;
        for (i = 0; i < chunks; ++i) {
            contexts[i].contextsLen = contextChunk + (i < contextAdd);
            contexts[i].contexts = aContextsGpu + contextOff;
            contextOff += contexts[i].contextsLen;
        }
        
        int cardsOff = 0;
        for (i = 0; i < chunks; ++i) {
        
            int cCardsLen = cardsChunk + (i < cardsAdd);
            int* cCards = cards + cardsOff;
            cardsOff += cCardsLen;

            for (j = 0; j < contexts[i].contextsLen; ++j) {
                contexts[i].contexts[j].cards = cCards;
                contexts[i].contexts[j].cardsLen = cCardsLen;
            }
        }
        
        for (i = 0; i < chunks; ++i) {
            aTasks[aContextsCpuLen + i] = threadPoolSubmit(alignsThread, &(contexts[i]));
        }
        
        for (i = 0; i < chunks; ++i) {
            threadPoolTaskWait(aTasks[aContextsCpuLen + i]);
            threadPoolTaskDelete(aTasks[aContextsCpuLen + i]);
        }

        free(contexts);
    }

    // wait for cpu tasks
    for (i = 0; i < aContextsCpuLen; ++i) {
        threadPoolTaskWait(aTasks[i]);
        threadPoolTaskDelete(aTasks[i]);
    }

    free(aContextsCpu);
    free(aContextsGpu);
    free(aTasks);
    
    TIMER_STOP;
    
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    for (i = 0; i < queriesLen; ++i) {
        free(dbAlignmentsData[i]);
    }
    free(dbAlignmentsData);

    //**************************************************************************
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// THREADS

static void* alignThread(void* param) {

    AlignContext* context = (AlignContext*) param;
    
    DbAlignment** dbAlignment = context->dbAlignment;
    int type = context->type;
    Chain* query = context->query;
    int queryIdx = context->queryIdx;
    Chain* target = context->target;
    int targetIdx = context->targetIdx;
    double value = context->value;
    int score = context->score;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    // align
    Alignment* alignment;
    alignPair(&alignment, type, query, target, scorer, cards, cardsLen, NULL);

    // check scores
    int s1 = alignmentGetScore(alignment);
    int s2 = score;

    ASSERT(s1 == s2, "Scores don't match %d %d, (%d %d)", s1, s2, queryIdx, targetIdx);

    // extract info
    int queryStart = alignmentGetQueryStart(alignment);
    int queryEnd = alignmentGetQueryEnd(alignment);
    int targetStart = alignmentGetTargetStart(alignment); 
    int targetEnd = alignmentGetTargetEnd(alignment);
    int pathLen = alignmentGetPathLen(alignment);

    char* path = (char*) malloc(pathLen);
    alignmentCopyPath(alignment, path);

    alignmentDelete(alignment);
        
    // create db alignment
    *dbAlignment = dbAlignmentCreate(query, queryStart, queryEnd, queryIdx, 
        target, targetStart, targetEnd, targetIdx, value, score, scorer, path, 
        pathLen);

    return NULL;
}

static void* alignsThread(void* param) {

    AlignContexts* context = (AlignContexts*) param;
    AlignContext* contexts = context->contexts;
    int contextsLen = context->contextsLen;
    
    int i = 0;
    for (i = 0; i < contextsLen; ++i) {
        alignThread(&(contexts[i]));
    }
    
    return NULL;
}

static void* extractThread(void* param) {

    ExtractContext* context = (ExtractContext*) param;
    
    DbAlignmentData** dbAlignmentData = context->dbAlignmentData;
    int* dbAlignmentLen = context->dbAlignmentLen;
    Chain* query = context->query;
    Chain** database = context->database;
    int databaseLen = context->databaseLen;
    int* scores = context->scores;
    int maxAlignments = context->maxAlignments;
    ValueFunction valueFunction = context->valueFunction;
    void* valueFunctionParam = context->valueFunctionParam;
    double valueThreshold = context->valueThreshold;
    
    int i;
    
    size_t packedSize = databaseLen * sizeof(DbAlignmentData);
    DbAlignmentData* packed = (DbAlignmentData*) malloc(packedSize);
    double* values = (double*) malloc(databaseLen * sizeof(double));
    
    valueFunction(values, scores, query, database, databaseLen, valueFunctionParam);

    int thresholded = 0;
    for (i = 0; i < databaseLen; ++i) {
    
        packed[i].idx = i;
        packed[i].value = values[i];
        packed[i].score = scores[i];
        
        if (packed[i].value <= valueThreshold) {
            thresholded++;
        }
    }
        
    int k = MIN(thresholded, maxAlignments);
    qselect((void*) packed, databaseLen, sizeof(DbAlignmentData), k, dbAlignmentDataCmp);
    qsort((void*) packed, k, sizeof(DbAlignmentData), dbAlignmentDataCmp);

    *dbAlignmentData = (DbAlignmentData*) malloc(k * sizeof(DbAlignmentData));
    *dbAlignmentLen = k;

    for (i = 0; i < k; ++i) {
        (*dbAlignmentData)[i] = packed[i];
    }

    free(packed);
    free(values);

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

static int dbAlignmentDataCmp(const void* a_, const void* b_) {

    DbAlignmentData* a = (DbAlignmentData*) a_;
    DbAlignmentData* b = (DbAlignmentData*) b_;
    
    if (a->value == b->value) {
        return b->score - a->score;
    }
    
    if (a->value < b->value) return -1;
    return 1;
}

//------------------------------------------------------------------------------
//******************************************************************************
