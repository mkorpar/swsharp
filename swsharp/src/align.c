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

typedef struct Context {
    Alignment** alignment;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int* cards;
    int cardsLen;
} Context;

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen, Thread* thread);
    
//******************************************************************************

//******************************************************************************
// PRIVATE

static void* hwAlignThread(void* param);

static void hwAlignGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void* nwAlignThread(void* param);

static void* swAlignThread(void* param);
  
static void swAlignSingleGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);

static void swAlignDualGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen);
    
static void swFindStartSpecific(int* queryStart, int* targetStart, 
    Chain* query, Chain* target, Scorer* scorer, int score, 
    int card, Thread* thread);
    
//******************************************************************************

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int type, int* cards, int cardsLen, Thread* thread) {
   
    Context* param = (Context*) malloc(sizeof(Context));

    param->alignment = alignment;
    param->query = query;
    param->target = target;
    param->scorer = scorer;
    param->cards = cards;
    param->cardsLen = cardsLen;

    void* (*function) (void*);
    
    switch (type) {
    case SW_ALIGN: 
        function = swAlignThread;
        break;
    case NW_ALIGN: 
        function = nwAlignThread;
        break;
    case HW_ALIGN: 
        function = hwAlignThread;
        break;
    default:
        ERROR("invalid align type");
    }
    
    if (thread == NULL) {
        function(param);
    } else {
        threadCreate(thread, function, (void*) param);
    } 
}

//******************************************************************************
    
//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// HW ALIGN

static void* hwAlignThread(void* param) {

    Context* context = (Context*) param;
    
    Alignment** alignment = context->alignment;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        alignPairCpu(alignment, HW_ALIGN, query, target, scorer);
    } else {
        hwAlignGpu(alignment, query, target, scorer, cards, cardsLen);
    }
    
    free(param);
    
    return NULL;
}

static void hwAlignGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {
    
    int card = cards[0];
    
    // find end and the score
    int queryEnd;
    int targetEnd;
    int score;

    hwEndDataGpu(&queryEnd, &targetEnd, &score, query, target, scorer, card, NULL);

    ASSERT(queryEnd == chainGetLength(query) - 1, "invalid hybrid alignment");

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

static void* nwAlignThread(void* param) {

    Context* context = (Context*) param;
    
    Alignment** alignment = context->alignment;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int score;
    char* path;
    int pathLen;
    
    nwReconstruct(&path, &pathLen, &score, query, 0, 0, target, 0, 0, 
        scorer, NO_SCORE, cards, cardsLen, NULL);
    
    *alignment = alignmentCreate(query, 0, rows - 1, target, 0, cols - 1, 
        score, scorer, path, pathLen);
    
    free(param);
    
    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SW ALIGN

static void* swAlignThread(void* param) {

    Context* context = (Context*) param;
    
    Alignment** alignment = context->alignment;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        alignPairCpu(alignment, SW_ALIGN, query, target, scorer);
    } else if (cardsLen == 1) {
        swAlignSingleGpu(alignment, query, target, scorer, cards, cardsLen);
    } else {
        swAlignDualGpu(alignment, query, target, scorer, cards, cardsLen);
    }    
    
    free(param);
    
    return NULL;
}

static void swAlignSingleGpu(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen) {

    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int score;

    swEndDataGpu(&queryEnd, &targetEnd, &score, NULL, NULL, query, target, 
        scorer, card, NULL);
  
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

static void swAlignDualGpu(Alignment** alignment, Chain* query, Chain* target, 
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
    for(up = 0, down = cols - 2; up < cols; ++up, --down) {
    
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
    
    int score = MAX(middleScore, MAX(upScore, downScore));

    LOG("Scores | up: %d | down: %d | mid: %d", upScore, downScore, middleScore);
    
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
        
    } else {
    
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
//******************************************************************************
