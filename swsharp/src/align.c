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

#include "alignment.h"
#include "chain.h"
#include "constants.h"
#include "cpu_module.h"
#include "error.h"
#include "gpu_module.h"
#include "reconstruct.h"
#include "scorer.h"
#include "utils.h"

#include "align.h"

#define GPU_MIN_LEN     256
#define GPU_MIN_CELLS   10000000.0

#define END_DATA_CPU        0
#define END_DATA_NW         1
#define END_DATA_HW         2
#define END_DATA_SW_SINGLE  3
#define END_DATA_SW_DUAL    4

typedef struct ContextPair {
    Alignment** alignment;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int type;
    int* cards;
    int cardsLen;
} ContextPair;

typedef struct ContextBest {
    Alignment** alignment;
    Chain** queries;
    int queriesLen;
    Chain* target;
    Scorer* scorer;
    int type;
    int* cards;
    int cardsLen;
} ContextBest;

typedef struct EndData {
    int type;
    int score;
    void* data;
} EndData;

typedef struct HwEndData {
    int queryEnd;
    int targetEnd;
} HwEndData;

typedef struct SwEndDataSingle {
    int queryEnd;
    int targetEnd;
} SwEndDataSingle;

typedef struct SwEndDataDual {
    int middleScore;
    int middleScoreUp;
    int middleScoreDown;
    int row;
    int col;
    int gap;
    int upScore;
    int upQueryEnd;
    int upTargetEnd;
    int downScore;
    int downQueryEnd;
    int downTargetEnd;
} SwEndDataDual;

typedef struct ScorePairContext {
    EndData** endData;
    Chain** queries;
    int queriesLen;
    Chain* target;
    Scorer* scorer;
    int type;
    int* cards;
    int cardsLen;
} ScorePairContext;

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen, Thread* thread);
    
extern void alignBest(Alignment** alignment, Chain** queries, int queriesLen, 
    Chain* target, Scorer* scorer, int type, int* cards, int cardsLen, 
    Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// pair
static void* alignPairThread(void* param);

extern void alignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen);

// best
static void* alignBestThread(void* param);

static void* scorePairThread(void* param);

static void scorePair(EndData** endData, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen);
    
static void reconstructPair(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int type, int* cards, 
    int cardsLen);

// hw
static void hwAlignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void hwScorePairGpu(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void hwReconstructPairGpu(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

// nw
static void nwAlignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void nwScorePairGpu(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void nwReconstructPairGpu(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

// sw
static void swAlignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void swScorePairGpu(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void swScorePairGpuSingle(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void swScorePairGpuDual(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);
    
static void swReconstructPairGpu(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

static void swReconstructPairGpuSingle(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

static void swReconstructPairGpuDual(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

static void swFindStartSpecific(int* queryStart, int* targetStart, 
    Chain* query, Chain* target, Scorer* scorer, int score,
    int card, Thread* thread);

// utils
static void deleteEndData(EndData* endData);
    
//******************************************************************************

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen, Thread* thread) {
   
    ContextPair* param = (ContextPair*) malloc(sizeof(ContextPair));

    param->alignment = alignment;
    param->query = query;
    param->target = target;
    param->scorer = scorer;
    param->type = type;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL) {
        alignPairThread(param);
    } else {
        threadCreate(thread, alignPairThread, (void*) param);
    }
}

extern void alignBest(Alignment** alignment, Chain** queries, int queriesLen, 
    Chain* target, Scorer* scorer, int type, int* cards, int cardsLen, 
    Thread* thread) {
    
    // reduce problem to simple pair align
    if (queriesLen == 1) {
        alignPair(alignment, queries[0], target, scorer, type, cards, 
            cardsLen, thread);
        return;
    }
    
    ContextBest* param = (ContextBest*) malloc(sizeof(ContextBest));

    param->alignment = alignment;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->target = target;
    param->scorer = scorer;
    param->type = type;
    param->cards = cards;
    param->cardsLen = cardsLen;
    
    if (thread == NULL) {
        alignBestThread(param);
    } else {
        threadCreate(thread, alignBestThread, (void*) param);
    }
}

//******************************************************************************
    
//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// ENTRY

static void* alignPairThread(void* param) {

    ContextPair* context = (ContextPair*) param;
    
    Alignment** alignment = context->alignment;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int type = context->type;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        alignPairCpu(alignment, type, query, target, scorer);
    } else {
        alignPairGpu(alignment, query, target, scorer, type, cards, cardsLen);
    }
    
    free(param);
    
    return NULL;
}

extern void alignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen) {

    void (*function) (Alignment**, Chain*, Chain*, Scorer*, int*, int);
    
    switch (type) {
    case HW_ALIGN:
        function = hwAlignPairGpu;
        break;
    case NW_ALIGN:
        function = nwAlignPairGpu;
        break;
    case SW_ALIGN:
        function = swAlignPairGpu;
        break;
    default:
        ERROR("invalid align type");
    }
    
    function(alignment, query, target, scorer, cards, cardsLen);
}

static void* alignBestThread(void* param) {

    ContextBest* context = (ContextBest*) param;

    Alignment** alignment = context->alignment;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int type = context->type;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    int i;
    
    //**************************************************************************
    // SCORE MULTITHREADED

    EndData** endData = (EndData**) malloc(queriesLen * sizeof(EndData*));
    
    int threadNmr = MIN(queriesLen, cardsLen);
    
    Thread* threads = 
        (Thread*) malloc((threadNmr - 1) * sizeof(Thread));
    
    ScorePairContext* contexts = 
        (ScorePairContext*) malloc(threadNmr * sizeof(ScorePairContext));

    int queriesStep = queriesLen / threadNmr;
    int cardsStep = cardsLen / threadNmr;

    for (i = 0; i < threadNmr; ++i) {
        contexts[i].endData = endData  + i * queriesStep;
        contexts[i].queries = queries + i * queriesStep;
        contexts[i].queriesLen = MIN(queriesStep, queriesLen - i * queriesStep);
        contexts[i].target = target;
        contexts[i].scorer = scorer;
        contexts[i].type = type;
        contexts[i].cards = cards + i * cardsStep;
        contexts[i].cardsLen = MIN(cardsStep, cardsLen - i * cardsStep);
    }
    
    for (i = 0; i < threadNmr - 1; ++i) {
        threadCreate(&threads[i], scorePairThread, &contexts[i]);
    }
    
    scorePairThread(&contexts[threadNmr - 1]);
    
    // wait for the threads
    for (i = 0; i < threadNmr - 1; ++i) {
        threadJoin(threads[i]);
    }

    //**************************************************************************
    
    //**************************************************************************
    // FIND AND ALIGN THE BEST
    
    int maxScore = SCORE_MIN;
    int index = -1;
    
    for (i = 0; i < queriesLen; ++i) {
    
        LOG("found %d", endData[i]->score);

        if (endData[i]->score > maxScore) {
            maxScore = endData[i]->score;
            index = i;
        }
    }
    
    reconstructPair(alignment, endData[index], queries[index], target, scorer, 
        type, cards, cardsLen);
    
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    for (i = 0; i < queriesLen; ++i) {
        deleteEndData(endData[i]);
    }
    free(endData);

    free(threads);
    free(contexts);
    
    free(param);
    
    //**************************************************************************
    
    return NULL;
}

static void* scorePairThread(void* param) {

    ScorePairContext* context = (ScorePairContext*) param;
    
    EndData** endData = context->endData;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int type = context->type;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int i;
    for (i = 0; i < queriesLen; ++i) {
        scorePair(&endData[i], queries[i], target, scorer, type, cards, cardsLen);
    }

    return NULL;
}

static void scorePair(EndData** endData, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen) {

    *endData = (EndData*) malloc(sizeof(EndData));
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        (*endData)->score = scorePairCpu(type, query, target, scorer);
        (*endData)->type = END_DATA_CPU;
        (*endData)->data = NULL;
        return;
    }
    
    void (*function) (EndData*, Chain*, Chain*, Scorer*, int*, int);
    
    switch (type) {
    case HW_ALIGN:
        function = hwScorePairGpu;
        break;
    case NW_ALIGN:
        function = nwScorePairGpu;
        break;
    case SW_ALIGN:
        function = swScorePairGpu;
        break;
    default:
        ERROR("invalid align type");
    }
    
    function(*endData, query, target, scorer, cards, cardsLen);
}

static void reconstructPair(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int type, int* cards, 
    int cardsLen) {

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    double cells = (double) rows * cols;
    
    if (endData->type == END_DATA_CPU || cardsLen == 0 || cols < GPU_MIN_LEN 
        || cells < GPU_MIN_CELLS) {
        alignPairCpu(alignment, type, query, target, scorer);
        return;
    }
    
    void (*function) (Alignment**, EndData*, Chain*, Chain*, Scorer*, int*, int);
    
    switch (type) {
    case HW_ALIGN:
        function = hwReconstructPairGpu;
        break;
    case NW_ALIGN:
        function = nwReconstructPairGpu;
        break;
    case SW_ALIGN:
        function = swReconstructPairGpu;
        break;
    default:
        ERROR("invalid align type");
    }
    
    function(alignment, endData, query, target, scorer, cards, cardsLen);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// HW ALIGN

static void hwAlignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {
    
    EndData* endData;
    scorePair(&endData, query, target, scorer, HW_ALIGN, cards, cardsLen);
    
    reconstructPair(alignment, endData, query, target, scorer, HW_ALIGN,
        cards, cardsLen);
    
    deleteEndData(endData);
}

static void hwScorePairGpu(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int score;

    hwEndDataGpu(&queryEnd, &targetEnd, &score, query, target, scorer, card, NULL);

    ASSERT(queryEnd == chainGetLength(query) - 1, "invalid hw alignment");
    
    HwEndData* data = (HwEndData*) malloc(sizeof(HwEndData));
    data->queryEnd = queryEnd;
    data->targetEnd = targetEnd;
    
    endData->type = END_DATA_HW;
    endData->score = score;
    endData->data = data;
}

static void hwReconstructPairGpu(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {

    ASSERT(endData->type == END_DATA_HW, "invalid end data type");

    int card = cards[0];
    
    int score = endData->score;
    
    // extract data
    HwEndData* data = (HwEndData*) endData->data;
    
    int queryEnd = data->queryEnd;
    int targetEnd = data->targetEnd;

    // find the start      
    Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
    Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);
  
    int* scores;
    nwLinearDataGpu(&scores, NULL, queryFind, 0, targetFind, 0, scorer, -1, -1, 
        card, NULL);
    
    int targetFindLen = chainGetLength(targetFind);
    
    chainDelete(queryFind);
    chainDelete(targetFind);
    
    int queryStart = 0;
    int targetStart = -1;
    
    int i;
    for (i = 0; i < targetFindLen; ++i) {
        if (scores[i] == score) {
            targetStart = targetFindLen - 1 - i;
            break;
        }
    }
   
    free(scores);
    
    ASSERT(targetStart != -1, "invalid hybrid find"); 

    // reconstruct
    char* path;
    int pathLen;
    
    Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
    Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
    nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
     
    chainDelete(queryRecn);
    chainDelete(targetRecn);  
         
    *alignment = alignmentCreate(query, queryStart, queryEnd, target, 
        targetStart, targetEnd, score, scorer, path, pathLen);
}
    
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// NW ALIGN

static void nwAlignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int score;
    char* path;
    int pathLen;
    
    nwReconstruct(&path, &pathLen, &score, query, 0, 0, target, 0, 0, 
        scorer, NO_SCORE, cards, cardsLen, NULL);
    
    *alignment = alignmentCreate(query, 0, rows - 1, target, 0, cols - 1, 
        score, scorer, path, pathLen);
}

static void nwScorePairGpu(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

    int* scores;
    
    nwLinearDataGpu(&scores, NULL, query, 0, target, 0, scorer, -1, -1, 
        cards[0], NULL);
        
    int score = scores[chainGetLength(target) - 1];
    free(scores);

    endData->type = END_DATA_NW;
    endData->score = score;
    endData->data = NULL;
}

static void nwReconstructPairGpu(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {

    ASSERT(endData->type == END_DATA_NW, "invalid end data type");

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int score = endData->score;

    int outScore;
    char* path;
    int pathLen;
    
    nwReconstruct(&path, &pathLen, &outScore, query, 0, 0, target, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
    
    ASSERT(score == outScore, "invalid nw align");
    
    *alignment = alignmentCreate(query, 0, rows - 1, target, 0, cols - 1, 
        score, scorer, path, pathLen);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SW ALIGN

static void swAlignPairGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

    EndData* endData;
    scorePair(&endData, query, target, scorer, SW_ALIGN, cards, cardsLen);
    
    reconstructPair(alignment, endData, query, target, scorer, SW_ALIGN,
        cards, cardsLen);
    
    deleteEndData(endData);
}

static void swScorePairGpu(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

    if (cardsLen > 1) {
        swScorePairGpuDual(endData, query, target, scorer, cards, cardsLen);
    } else {
        swScorePairGpuSingle(endData, query, target, scorer, cards, cardsLen);
    }
}

static void swScorePairGpuSingle(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {
    
    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int score;

    swEndDataGpu(&queryEnd, &targetEnd, &score, NULL, NULL, query, target, 
        scorer, card, NULL);

    SwEndDataSingle* data = (SwEndDataSingle*) malloc(sizeof(SwEndDataSingle));
    data->queryEnd = queryEnd;
    data->targetEnd = targetEnd;
    
    endData->score = score;
    endData->data = data;
    endData->type = END_DATA_SW_SINGLE;
}

static void swScorePairGpuDual(EndData* endData, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {
    
    Thread thread;
        
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int row = rows / 2;

    int* upScores; 
    int* upAffines;
    int upQueryEnd;
    int upTargetEnd;
    int upScore;
    Chain* upRow = chainCreateView(query, 0, row, 0);
    Chain* upCol = chainCreateView(target, 0, cols - 1, 0);

    int* downScores; 
    int* downAffines;
    int downQueryEnd;
    int downTargetEnd;
    int downScore;
    Chain* downRow = chainCreateView(query, row + 1, rows - 1, 1);
    Chain* downCol = chainCreateView(target, 0, cols - 1, 1);

    if (cardsLen == 1) {
    
        swEndDataGpu(&upQueryEnd, &upTargetEnd, &upScore, &upScores, &upAffines,
            upRow, upCol, scorer, cards[0], NULL);
            
        swEndDataGpu(&downQueryEnd, &downTargetEnd, &downScore, &downScores, 
            &downAffines, downRow, downCol, scorer, cards[0], NULL);
            
    } else {
    
        swEndDataGpu(&upQueryEnd, &upTargetEnd, &upScore, &upScores, &upAffines,
            upRow, upCol, scorer, cards[1], &thread);
            
        swEndDataGpu(&downQueryEnd, &downTargetEnd, &downScore, &downScores, 
            &downAffines, downRow, downCol, scorer, cards[0], NULL);
            
        threadJoin(thread); 
    }
    
    chainDelete(upRow);
    chainDelete(upCol);
    chainDelete(downCol);
    chainDelete(downRow);
    
    int middleScore = INT_MIN;
    int middleScoreUp = 0;
    int middleScoreDown = 0;
    int gap = 0; // boolean
    int col = -1;

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    
    int up, down;
    for(up = 0, down = cols - 2; up < cols - 1; ++up, --down) {
    
        int scr = upScores[up] + downScores[down];
        int aff = upAffines[up] + downAffines[down] + gapOpen - gapExtend;
        
        int isScrAff = (upScores[up] == upAffines[up]) && 
                       (downScores[down] ==  downAffines[down]);
        
        if (scr > middleScore || (scr == middleScore && !isScrAff)) {
            middleScoreUp = upScores[up];
            middleScoreDown = downScores[down];
            middleScore = scr;
            gap = 0;
            col = up;   
        }

        if (aff >= middleScore) {
            middleScoreUp = upAffines[up];
            middleScoreDown = downAffines[down];
            middleScore = aff;
            gap = 1;
            col = up;   
        }
    }
    
    free(upScores);
    free(upAffines);
    free(downScores);
    free(downAffines);

    LOG("Scores | up: %d | down: %d | mid: %d", upScore, downScore, middleScore);
    
    int score = MAX(middleScore, MAX(upScore, downScore));
    
    SwEndDataDual* data = (SwEndDataDual*) malloc(sizeof(SwEndDataDual));
    data->middleScore = middleScore;
    data->middleScoreUp = middleScoreUp;
    data->middleScoreDown = middleScoreDown;
    data->row = row;
    data->col = col;
    data->gap = gap;
    data->upScore = upScore;
    data->upQueryEnd = upQueryEnd;
    data->upTargetEnd = upTargetEnd;
    data->downScore = downScore;
    data->downQueryEnd = downQueryEnd;
    data->downTargetEnd = downTargetEnd;
    
    endData->score = score;
    endData->data = data;
    endData->type = END_DATA_SW_DUAL;
}

static void swReconstructPairGpu(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {

    if (endData->type == END_DATA_SW_DUAL) {
        swReconstructPairGpuDual(alignment, endData, query, target, scorer, 
            cards, cardsLen);
    } else {
        swReconstructPairGpuSingle(alignment, endData, query, target, scorer, 
            cards, cardsLen);
    }
}

static void swReconstructPairGpuSingle(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
    
    ASSERT(endData->type == END_DATA_SW_SINGLE, "invalid end data type");
    
    int card = cards[0];
    
    int score = endData->score;
    
    // extract data
    SwEndDataSingle* data = (SwEndDataSingle*) endData->data;
    
    int queryEnd = data->queryEnd;
    int targetEnd = data->targetEnd;
    
    Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
    Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);

    int queryStart;
    int targetStart;

    swFindStartSpecific(&queryStart, &targetStart, queryFind, targetFind, 
        scorer, score, card, NULL);

    queryStart = chainGetLength(queryFind) - queryStart - 1;
    targetStart = chainGetLength(targetFind) - targetStart - 1;

    chainDelete(queryFind);
    chainDelete(targetFind);
    
    Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
    Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
    int pathLen;
    char* path;
    
    nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
    
    chainDelete(queryRecn);
    chainDelete(targetRecn);
    
    *alignment = alignmentCreate(query, queryStart, queryEnd, target, 
        targetStart, targetEnd, score, scorer, path, pathLen);
}

static void swReconstructPairGpuDual(Alignment** alignment, EndData* endData, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
    
    ASSERT(endData->type == END_DATA_SW_DUAL, "invalid end data type");
    
    int score = endData->score;
    
    // extract data
    SwEndDataDual* data = (SwEndDataDual*) endData->data;
    int middleScore = data->middleScore;
    int middleScoreUp = data->middleScoreUp;
    int middleScoreDown = data->middleScoreDown;
    int row = data->row;
    int col = data->col;
    int gap = data->gap;
    int upScore = data->upScore;
    int upQueryEnd = data->upQueryEnd;
    int upTargetEnd = data->upTargetEnd;
    int downScore = data->downScore;
    int downQueryEnd = data->downQueryEnd;
    int downTargetEnd = data->downTargetEnd;
    
    Thread thread;
        
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int queryEnd;
    int targetEnd;
    
    int queryStart;
    int targetStart;
    
    int pathLen;
    char* path;
    
    if (score == middleScore) { 
    
        Chain* upQueryFind = chainCreateView(query, 0, row, 1);
        Chain* upTargetFind = chainCreateView(target, 0, col, 1);
       
        Chain* downQueryFind = chainCreateView(query, row + 1, rows - 1, 0);
        Chain* downTargetFind = chainCreateView(target, col + 1, cols - 1, 0);
        
        if (cardsLen == 1) {
        
            swFindStartSpecific(&queryStart, &targetStart,  upQueryFind, 
                upTargetFind, scorer, middleScoreUp, cards[0], NULL);
                
            swFindStartSpecific(&queryEnd, &targetEnd,  downQueryFind, 
                downTargetFind, scorer, middleScoreDown, cards[0], NULL);
        
        } else {
        
            swFindStartSpecific(&queryStart, &targetStart, upQueryFind, 
                upTargetFind, scorer, middleScoreUp, cards[1], &thread);
                
            swFindStartSpecific(&queryEnd, &targetEnd,  downQueryFind, 
                downTargetFind, scorer, middleScoreDown, cards[0], NULL);
                
            threadJoin(thread);
        }
        
        chainDelete(upQueryFind);
        chainDelete(upTargetFind);
        chainDelete(downQueryFind);
        chainDelete(downTargetFind);
        
        queryStart = row - queryStart;
        targetStart = col - targetStart;
        queryEnd += row + 1;
        targetEnd += col + 1;
        
        char* upPath;
        int upPathLen;
        Chain* upQueryRecn = chainCreateView(query, queryStart, row, 0);
        Chain* upTargetRecn = chainCreateView(target, targetStart, col, 0);
        
        char* downPath;
        int downPathLen;
        Chain* downQueryRecn = chainCreateView(query, row + 1, queryEnd, 0);
        Chain* downTargetRecn = chainCreateView(target, col + 1, targetEnd, 0);
        
        if (cardsLen == 1) {
        
            nwReconstruct(&upPath, &upPathLen, NULL, upQueryRecn, 0, gap, 
                upTargetRecn, 0, 0, scorer, middleScoreUp, cards, cardsLen, NULL);
                
            nwReconstruct(&downPath, &downPathLen, NULL, downQueryRecn, gap, 0, 
                downTargetRecn, 0, 0, scorer, middleScoreDown, 
                cards, cardsLen, NULL);
        
        } else {
        
            int half = cardsLen / 2;
            
            nwReconstruct(&upPath, &upPathLen, NULL, upQueryRecn, 0, gap, 
                upTargetRecn, 0, 0, scorer, middleScoreUp, cards, half, &thread);
                
            nwReconstruct(&downPath, &downPathLen, NULL, downQueryRecn, gap, 0, 
                downTargetRecn, 0, 0, scorer, middleScoreDown, cards + half, 
                cardsLen - half, NULL);
                
            threadJoin(thread);
        }
        
        chainDelete(upQueryRecn);
        chainDelete(upTargetRecn);
        chainDelete(downQueryRecn);
        chainDelete(downTargetRecn);

        pathLen = upPathLen + downPathLen;
        path = (char*) malloc(pathLen * sizeof(char));

        memcpy(path, upPath, upPathLen * sizeof(char));
        memcpy(path + upPathLen, downPath, downPathLen * sizeof(char));
                
        free(upPath);
        free(downPath);
        
    } else if (score == upScore) {
    
        queryEnd = upQueryEnd;
        targetEnd = upTargetEnd;
        
        Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
        Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);
        
        swFindStartSpecific(&queryStart, &targetStart, queryFind, targetFind, 
            scorer, score, cards[0], NULL);
            
        queryStart = chainGetLength(queryFind) - queryStart - 1;
        targetStart = chainGetLength(targetFind) - targetStart - 1;
        
        chainDelete(queryFind);
        chainDelete(targetFind);
        
        Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
        Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
        nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0, 
            scorer, score, cards, cardsLen, NULL);
            
        chainDelete(queryRecn);
        chainDelete(targetRecn);
        
    } else if (score == downScore) {
    
        queryStart = chainGetLength(query) - downQueryEnd - 1;
        targetStart = chainGetLength(target) - downTargetEnd - 1;
        
        Chain* queryFind = chainCreateView(query, queryStart, rows - 1, 0);
        Chain* targetFind = chainCreateView(target, targetStart, cols - 1, 0);
        
        swFindStartSpecific(&queryEnd, &targetEnd, queryFind, targetFind, 
            scorer, score, cards[0], NULL);
            
        queryEnd = queryStart + queryEnd;
        targetEnd = targetStart + targetEnd;
        
        chainDelete(queryFind);
        chainDelete(targetFind);
        
        Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
        Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
        nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0, 
            scorer, score, cards, cardsLen, NULL);
            
        chainDelete(queryRecn);
        chainDelete(targetRecn);
    } else {
        ERROR("invalid dual data score");
    }
    
    *alignment = alignmentCreate(query, queryStart, queryEnd, target, 
        targetStart, targetEnd, score, scorer, path, pathLen);
}

static void swFindStartSpecific(int* queryStart, int* targetStart, 
    Chain* query, Chain* target, Scorer* scorer, int score,
    int card, Thread* thread) {
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS) {
        swFindStartCpu(queryStart, targetStart, query, target, scorer, score);
    } else {
        swFindStartGpu(queryStart, targetStart, query, target, scorer, score, 
            card, thread);
    }
    
    ASSERT(*queryStart != -1, "Score not found %d", score);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// UTILS

static void deleteEndData(EndData* endData) {
    free(endData->data);
    free(endData);
    endData = NULL;
}

//------------------------------------------------------------------------------
//******************************************************************************
