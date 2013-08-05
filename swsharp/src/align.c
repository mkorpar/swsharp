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
#define GPU_MIN_CELLS   1000000.0

typedef struct ContextBest {
    Alignment** alignment;
    int type;
    Chain** queries;
    int queriesLen;
    Chain* target;
    Scorer* scorer;
    int* cards;
    int cardsLen;
} ContextBest;

typedef struct ContextPair {
    Alignment** alignment;
    int type;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int* cards;
    int cardsLen;
} ContextPair;

typedef struct ContextScore {
    int* score;
    void** data;
    int type;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int* cards;
    int cardsLen;
} ContextScore;

typedef struct ContextPairs {
    ContextScore** contexts;
    int contextsLen;
    int offset;
    int step;
    int* cards;
    int cardsLen;
} ContextPairs;

typedef struct HwData {
    int score;
    int queryEnd;
    int targetEnd;
} HwData;

typedef struct NwData {
    int score;
} NwData;

typedef struct SwDataSingle {
    int score;
    int queryEnd;
    int targetEnd;
} SwDataSingle;

typedef struct SwDataDual {
    int score;
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
} SwDataDual;

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int* cards, int cardsLen, Thread* thread);

extern void alignBest(Alignment** alignment, int type, Chain** queries, 
    int queriesLen, Chain* target, Scorer* scorer, int* cards, int cardsLen, 
    Thread* thread);

extern void scorePair(int* score, int type, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen, Thread* thread);
    
//******************************************************************************

//******************************************************************************
// PRIVATE

static void* alignPairThread(void* param);

static void* alignBestThread(void* param);

static void* scorePairThread(void* param);

static void* scorePairsThread(void* param);

static int scorePairGpu(void** data, int type, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);
    
static void reconstructPairGpu(Alignment** alignment, void* data, int type, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

// hw
static int hwScorePairGpu(void** data, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);
    
static void hwReconstructPairGpu(Alignment** alignment, void* data, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
// nw
static int nwScorePairGpu(void** data, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void nwReconstructPairGpu(Alignment** alignment, void* data, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
static void nwFindScoreSpecific(int* queryStart, int* targetStart, Chain* query, 
    Chain* target, Scorer* scorer, int score, int card, Thread* thread);
    
// sw
static int swScorePairGpuSingle(void** data, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void swReconstructPairGpuSingle(Alignment** alignment, void* data,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
static int swScorePairGpuDual(void** data, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void swReconstructPairGpuDual(Alignment** alignment, void* data, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
//******************************************************************************

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int* cards, int cardsLen, Thread* thread) {
   
    ContextPair* param = (ContextPair*) malloc(sizeof(ContextPair));

    param->alignment = alignment;
    param->type = type;
    param->query = query;
    param->target = target;
    param->scorer = scorer;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL) {
        alignPairThread(param);
    } else {
        threadCreate(thread, alignPairThread, (void*) param);
    }
}

extern void alignBest(Alignment** alignment, int type, Chain** queries, 
    int queriesLen, Chain* target, Scorer* scorer, int* cards, int cardsLen, 
    Thread* thread) {
    
    // reduce problem to simple pair align
    if (queriesLen == 1) {
        alignPair(alignment, type, queries[0], target, scorer, cards, 
            cardsLen, thread);
        return;
    }
    
    ContextBest* param = (ContextBest*) malloc(sizeof(ContextBest));

    param->alignment = alignment;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->target = target;
    param->scorer = scorer;
    param->cards = cards;
    param->cardsLen = cardsLen;
    
    if (thread == NULL) {
        alignBestThread(param);
    } else {
        threadCreate(thread, alignBestThread, (void*) param);
    }
}

extern void scorePair(int* score, int type, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen, Thread* thread) {
    
    ContextScore* param = (ContextScore*) malloc(sizeof(ContextScore));

    param->score = score;
    param->data = NULL; // not needed
    param->type = type;
    param->query = query;
    param->target = target;
    param->scorer = scorer;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL) {
        scorePairThread(param);
    } else {
        threadCreate(thread, scorePairThread, (void*) param);
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
    int type = context->type;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        alignPairCpu(alignment, type, query, target, scorer);
    } else {
        
        void* data;
        scorePairGpu(&data, type, query, target, scorer, cards, cardsLen);

        reconstructPairGpu(alignment, data, type, query, target, scorer, 
            cards, cardsLen);
            
        free(data);        
    }
    
    free(param);
    
    return NULL;
}

static void* alignBestThread(void* param) {

    ContextBest* context = (ContextBest*) param;

    Alignment** alignment = context->alignment;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    int i;
    
    //**************************************************************************
    // SCORE MULTITHREADED
    
    int threadNmr = MIN(queriesLen, cardsLen);

    Thread* threads = (Thread*) malloc((threadNmr - 1) * sizeof(Thread));

    void** data = (void**) malloc(queriesLen * sizeof(void*));
    int* scores = (int*) malloc(queriesLen * sizeof(int));
    
    size_t contextScoresSize = queriesLen * sizeof(ContextScore);
    ContextScore** contextScores = (ContextScore**) malloc(contextScoresSize);

    for (i = 0; i < queriesLen; ++i) {
    
        contextScores[i] = (ContextScore*) malloc(sizeof(ContextScore));
        
        contextScores[i]->score = &(scores[i]);
        contextScores[i]->data = &(data[i]);;
        contextScores[i]->type = type;
        contextScores[i]->query = queries[i];
        contextScores[i]->target = target;
        contextScores[i]->scorer = scorer;
    }
    
    size_t contextSize = threadNmr * sizeof(ContextPairs);
    ContextPairs** contexts = (ContextPairs**) malloc(contextSize);
    
    int cardsStep = cardsLen / threadNmr;
    
    for (i = 0; i < threadNmr; ++i) {
    
        contexts[i] = (ContextPairs*) malloc(sizeof(ContextPairs));
        
        contexts[i]->contexts = contextScores;
        contexts[i]->contextsLen = queriesLen;
        contexts[i]->offset = i;
        contexts[i]->step = threadNmr;
        contexts[i]->cards = cards + i * cardsStep;

        if (i == threadNmr - 1) {
            // possible leftovers
            contexts[i]->cardsLen = cardsLen - i * cardsStep;
        } else {
            contexts[i]->cardsLen = cardsStep;
        }
    }

    for (i = 0; i < threadNmr - 1; ++i) {
        threadCreate(&threads[i], scorePairsThread, contexts[i]);
    }
    
    scorePairsThread(contexts[threadNmr - 1]);
    
    // wait for the threads
    for (i = 0; i < threadNmr - 1; ++i) {
        threadJoin(threads[i]);
    }

    //**************************************************************************

    //**************************************************************************
    // FIND AND ALIGN THE BEST

    int maxScore = scores[0];
    int maxIdx = 0;
    
    for (i = 1; i < queriesLen; ++i) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxIdx = i;
        }
    }
    
    reconstructPairGpu(alignment, data[maxIdx], type, queries[maxIdx], target, 
        scorer, cards, cardsLen);
    
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    for (i = 0; i < queriesLen; ++i) {
        free(data[i]);
    }
    
    free(data);
    free(scores);
    free(contextScores);

    free(contexts);
    free(threads);
    
    free(param);

    //**************************************************************************
    
    return NULL;
}

static void* scorePairThread(void* param) {

    ContextScore* context = (ContextScore*) param;

    int* score = context->score;
    void** data = context->data;
    int type = context->type;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        *score = scorePairCpu(type, query, target, scorer);
    } else {
        *score = scorePairGpu(data, type, query, target, scorer, cards, cardsLen);
    }
    
    free(param);
        
    return NULL;
}

static void* scorePairsThread(void* param) {

    ContextPairs* context = (ContextPairs*) param;
    
    ContextScore** contexts = context->contexts;
    int contextsLen = context->contextsLen;
    int offset = context->offset;
    int step = context->step;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    int i;
    for (i = offset; i < contextsLen; i += step) {

        // write own card information
        contexts[i]->cards = cards;
        contexts[i]->cardsLen = cardsLen;
        
        scorePairThread(contexts[i]);
    }
    
    free(param);
    
    return NULL;
}

static int scorePairGpu(void** data, int type, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

    int dual = cardsLen >= 2;
    
    int (*function) (void**, Chain*, Chain*, Scorer*, int*, int);
    
    switch (type) {
    case HW_ALIGN:
        function = hwScorePairGpu;
        break;
    case NW_ALIGN:
        function = nwScorePairGpu;
        break;
    case SW_ALIGN:
        if (dual) {
            function = swScorePairGpuDual;
        } else {
            function = swScorePairGpuSingle;
        }
        break;
    default:
        ERROR("invalid align type");
    }
    
    return function(data, query, target, scorer, cards, cardsLen);
}
    
static void reconstructPairGpu(Alignment** alignment, void* data, int type, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {

    int dual = cardsLen >= 2;

    void (*function) (Alignment**, void*, Chain*, Chain*, Scorer*, int*, int);
    
    switch (type) {
    case HW_ALIGN:
        function = hwReconstructPairGpu;
        break;
    case NW_ALIGN:
        function = nwReconstructPairGpu;
        break;
    case SW_ALIGN:
        if (dual) {
            function = swReconstructPairGpuDual;
        } else {
            function = swReconstructPairGpuSingle;
        }
        break;
    default:
        ERROR("invalid align type");
    }
    
    function(alignment, data, query, target, scorer, cards, cardsLen);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// hw

static int hwScorePairGpu(void** data_, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {
    
    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int score;

    hwEndDataGpu(&queryEnd, &targetEnd, &score, query, target, scorer, card, NULL);

    ASSERT(queryEnd == chainGetLength(query) - 1, "invalid hw alignment");
    
    if (data_ != NULL) {
    
        HwData* data = (HwData*) malloc(sizeof(HwData));
        data->score = score;
        data->queryEnd = queryEnd;
        data->targetEnd = targetEnd;
        
        *data_ = data;
    }

    return score;
}
    
static void hwReconstructPairGpu(Alignment** alignment, void* data_, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
    
    HwData* data = (HwData*) data_;
    
    int score = data->score;
    int queryEnd = data->queryEnd;
    int targetEnd = data->targetEnd;
    
    int card = cards[0];
    
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
// nw

static int nwScorePairGpu(void** data_, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {
    
    int* scores;
    
    nwLinearDataGpu(&scores, NULL, query, 0, target, 0, scorer, -1, -1, 
        cards[0], NULL);
        
    int score = scores[chainGetLength(target) - 1];
    free(scores);

    if (data_ != NULL) {
    
        NwData* data = (NwData*) malloc(sizeof(NwData));
        data->score = score;
        
        *data_ = data;
    }
    
    return score;
}

static void nwReconstructPairGpu(Alignment** alignment, void* data_,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
    
    NwData* data = (NwData*) data_;
    int score = data->score;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    char* path;
    int pathLen;
    
    nwReconstruct(&path, &pathLen, &score, query, 0, 0, target, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
    
    *alignment = alignmentCreate(query, 0, rows - 1, target, 0, cols - 1, 
        score, scorer, path, pathLen);
}

static void nwFindScoreSpecific(int* queryStart, int* targetStart, Chain* query, 
    Chain* target, Scorer* scorer, int score, int card, Thread* thread) {
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS) {
        nwFindScoreCpu(queryStart, targetStart, query, target, scorer, score);
    } else {
        nwFindScoreGpu(queryStart, targetStart, query, target, scorer, score, 
            card, thread);
    }
    
    ASSERT(*queryStart != -1, "Score not found %d", score);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SW

static int swScorePairGpuSingle(void** data_, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {
    
    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int score;

    swEndDataGpu(&queryEnd, &targetEnd, &score, NULL, NULL, query, target, 
        scorer, card, NULL);

    if (data_ != NULL) {
    
        SwDataSingle* data = (SwDataSingle*) malloc(sizeof(SwDataSingle));
        data->score = score;
        data->queryEnd = queryEnd;
        data->targetEnd = targetEnd;
        
        *data_ = data;
    }
    
    return score;
}

static void swReconstructPairGpuSingle(Alignment** alignment, void* data_,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
  
    SwDataSingle* data = (SwDataSingle*) data_;
    int score = data->score;
    int queryEnd = data->queryEnd;
    int targetEnd = data->targetEnd;
    
    if (score == 0) {
        *alignment = alignmentCreate(query, 0, 0, target, 0, 0, 0, scorer, NULL, 0);
        return;
    }

    int card = cards[0];
    
    Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
    Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);

    int queryStart;
    int targetStart;

    nwFindScoreSpecific(&queryStart, &targetStart, queryFind, targetFind, 
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

static int swScorePairGpuDual(void** data_, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

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
    
        Thread thread;
        
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
    
    if (upScore == 0 || downScore == 0) {
        *data_ = NULL;
        return 0;
    }
    
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

        if (scr > middleScore) {
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

    int score = MAX(middleScore, MAX(upScore, downScore));
    
    LOG("Scores | up: %d | down: %d | mid: %d", upScore, downScore, middleScore);
    
    if (data_ != NULL) {
    
        SwDataDual* data = (SwDataDual*) malloc(sizeof(SwDataDual));
        data->score = score;
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
        
        *data_ = data;
    }
    
    return score;
}

static void swReconstructPairGpuDual(Alignment** alignment, void* data_,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {

    if (data_ == NULL) {
        *alignment = alignmentCreate(query, 0, 0, target, 0, 0, 0, scorer, NULL, 0);
        return;
    }

    // extract data
    SwDataDual* data = (SwDataDual*) data_;
    int score = data->score;
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
        
            nwFindScoreSpecific(&queryStart, &targetStart,  upQueryFind, 
                upTargetFind, scorer, middleScoreUp, cards[0], NULL);
                
            nwFindScoreSpecific(&queryEnd, &targetEnd,  downQueryFind, 
                downTargetFind, scorer, middleScoreDown, cards[0], NULL);
        
        } else {
        
            nwFindScoreSpecific(&queryStart, &targetStart, upQueryFind, 
                upTargetFind, scorer, middleScoreUp, cards[1], &thread);
                
            nwFindScoreSpecific(&queryEnd, &targetEnd,  downQueryFind, 
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
        
        nwFindScoreSpecific(&queryStart, &targetStart, queryFind, targetFind, 
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
        
        nwFindScoreSpecific(&queryEnd, &targetEnd, queryFind, targetFind, 
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
    
//------------------------------------------------------------------------------
//******************************************************************************
