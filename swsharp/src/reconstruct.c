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
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "chain.h"
#include "cpu_module.h"
#include "constants.h"
#include "error.h"
#include "gpu_module.h"
#include "scorer.h"
#include "thread.h"
#include "utils.h"

#include "reconstruct.h"

#define MIN_DUAL_LEN        20000
#define MIN_BLOCK_SIZE      256 // > MINIMAL THREADS IN LINEAR_DATA * 2 !!!!
#define MAX_BLOCK_CELLS     5000000.0
#define THREADS             4

typedef struct Block {
    int outScore;
    int score;
    int queryFrontGap; // boolean
    int queryBackGap; // boolean
    int targetFrontGap; // boolean
    int targetBackGap; // boolean
    int startRow;
    int startCol;
    int endRow;
    int endCol;
    char* path;
    int pathLen;
} Block;

typedef struct Context {
    char** path;
    int* pathLen;
    int* outScore;
    Chain* query; 
    int queryFrontGap; // boolean
    int queryBackGap; // boolean
    Chain* target; 
    int targetFrontGap; // boolean
    int targetBackGap; // boolean
    Scorer* scorer; 
    int score;
    int* cards;
    int cardsLen;
} Context; 

typedef struct Queue {
    Semaphore mutex;
    Semaphore write;
    Semaphore done;
    Block** data;
    int length;
    int current;
} Queue;

typedef struct QueueContext {
    Queue* queue;
    Chain* query;
    Chain* target;
    Scorer* scorer;
} QueueContext;

typedef struct HirschbergContext {
    Block* block;
    Queue* queue;
    Chain* rowChain;
    Chain* colChain;
    Scorer* scorer; 
    int* cards;
    int cardsLen;
} HirschbergContext;

//******************************************************************************
// PUBLIC

extern void nwReconstruct(char** path, int* pathLen, int* outScore, 
    Chain* query, int queryFrontGap, int queryBackGap, Chain* target, 
    int targetFrontGap, int targetBackGap, Scorer* scorer, int score, 
    int* cards, int cardsLen, Thread* thread);

//******************************************************************************
        
//******************************************************************************
// PRIVATE 
               
static void* nwReconstructThread(void* param);
static void* hirschberg(void* param);

static void queueAdd(Queue* queue, Block* block);
static void queueCreate(Queue* queue, int size);
static void queueDelete(Queue* queue);
static void* queueWorker(void* params);

static int blockCmp(const void* a_, const void* b_);

//******************************************************************************

extern void nwReconstruct(char** path, int* pathLen, int* outScore, 
    Chain* query, int queryFrontGap, int queryBackGap, Chain* target, 
    int targetFrontGap, int targetBackGap, Scorer* scorer, int score, 
    int* cards, int cardsLen, Thread* thread) {
    
    Context* param = (Context*) malloc(sizeof(Context));

    param->path = path;
    param->pathLen = pathLen;
    param->outScore = outScore;
    param->query = query;
    param->queryFrontGap = queryFrontGap;
    param->queryBackGap = queryBackGap;
    param->target = target;
    param->targetFrontGap = targetFrontGap;
    param->targetBackGap = targetBackGap;
    param->scorer = scorer;
    param->score = score;
    param->cards = cards;
    param->cardsLen = cardsLen;
    
    if (thread == NULL) {
        nwReconstructThread(param);
    } else {
        threadCreate(thread, nwReconstructThread, (void*) param);
    }
}

//******************************************************************************
        
//******************************************************************************
// PRIVATE 

//------------------------------------------------------------------------------
// RECONSTRUCT

static void* nwReconstructThread(void* param) {
   
    Context* context = (Context*) param;
    
    char** path = context->path;
    int* pathLen = context->pathLen;
    int* outScore = context->outScore;
    Chain* query = context->query; 
    int queryFrontGap = context->queryFrontGap;
    int queryBackGap = context->queryBackGap; 
    Chain* target = context->target; 
    int targetFrontGap = context->targetFrontGap;
    int targetBackGap = context->targetBackGap; 
    Scorer* scorer = context->scorer; 
    int score = context->score;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    TIMER_START("Reconstruction");
     
    //**************************************************************************
    // PARALLEL CPU/GPU RECONSTRUCTION 

    Queue queue;
    queueCreate(&queue, 1 + (MAX(rows, cols) * 2) / MIN_BLOCK_SIZE);
    
    QueueContext queueContext = { &queue, query, target, scorer };
    
    Thread threads[THREADS];
    int threadIdx;   
    for (threadIdx = 0; threadIdx < THREADS; ++threadIdx) {
        threadCreate(&(threads[threadIdx]), queueWorker, (void*) &queueContext);
    }
    
    Block* topBlock = (Block*) malloc(sizeof(Block));
    topBlock->score = score;
    topBlock->queryFrontGap = queryFrontGap;
    topBlock->queryBackGap = queryBackGap;
    topBlock->targetFrontGap = targetFrontGap;
    topBlock->targetBackGap = targetBackGap;
    topBlock->startRow = 0;
    topBlock->startCol = 0;
    topBlock->endRow = rows - 1;
    topBlock->endCol = cols - 1;
    
    HirschbergContext hirschbergContext = { topBlock, &queue, query, target,
        scorer, cards, cardsLen };
    
    // TIMER_START("Hirschberg");
    // WARNING : topBlock is deleted in this function
    hirschberg(&hirschbergContext);
    // TIMER_STOP;
    
    // TIMER_START("CPU reconstrcution overhead");
    
    // wait until empty semaphore gets equal 0, until all of the cpu threads
    // finish
    while (1) { 

        if (semaphoreValue(&(queue.done)) == queue.length) {
            break;
        }
        
        threadSleep(10);
    }
    
    // threads are finished and stuck, cancel them
    for (threadIdx = 0; threadIdx < THREADS; ++threadIdx) {
        threadCancel(threads[threadIdx]);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CONCATENATE THE RESULT 
    
    Block** blocks = queue.data;
    int blocksLen = queue.length;
    int blockIdx;
    
    // becouse of multithreading blocks may not be in order
    qsort(blocks, blocksLen, sizeof(Block*), blockCmp);
    
    *pathLen = 0;
    for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx) {
        *pathLen += blocks[blockIdx]->pathLen;
    }

    *path = (char*) malloc(*pathLen * sizeof(char));
    char* pathPtr = *path;
    
    for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx) {
        size_t size = blocks[blockIdx]->pathLen * sizeof(char);
        memcpy(pathPtr, blocks[blockIdx]->path, size);
        pathPtr += blocks[blockIdx]->pathLen;
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CALCULATE OUT SCORE
    
    if (outScore !=  NULL) {
        
        int gapDiff = scorerGetGapOpen(scorer) - scorerGetGapExtend(scorer);
    
        *outScore = 0;
    
        for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx) {
        
            Block* block = blocks[blockIdx];
            
            *outScore += block->outScore;
            
            // if two consecutive blocks are connected with a gap compensate the
            // double gap opening penalty, use only the back gaps since there
            // needs to be only one compensation per gap
            *outScore += block->queryFrontGap * gapDiff;
            *outScore += block->targetBackGap * gapDiff;
        }
    }
    
    // TIMER_STOP;
    
    //**************************************************************************    
        
    queueDelete(&queue);
    
    free(param);
    
    TIMER_STOP;
    
    return NULL;
}

static void* hirschberg(void* param) {

    HirschbergContext* context = (HirschbergContext*) param;
    
    Block* block = context->block;
    Queue* queue = context->queue;
    Chain* rowChain = context->rowChain;
    Chain* colChain = context->colChain;
    Scorer* scorer = context->scorer; 
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(rowChain);
    int cols = chainGetLength(colChain);
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int gapDiff = gapOpen - gapExtend;
    
    int t = MAX(rows, cols) - block->score / scorerGetMaxScore(scorer);
    int p = (t - abs(rows - cols)) / 2;

    double cells = (double) (2 * p + abs(rows - cols) + 1) * cols;
    
    if (rows < MIN_BLOCK_SIZE || cols < MIN_BLOCK_SIZE || 
        cells < MAX_BLOCK_CELLS || cardsLen == 0) {
        
        // mm algorithm compensates and finds subblocks which often do not need
        // to have the optimal aligment, therefore it compensates the non 
        // optimal aligment by reducing gap penalties, these have to be 
        // compensated back
        block->score -= block->queryFrontGap * gapDiff;
        block->score -= block->queryBackGap * gapDiff;
        block->score -= block->targetFrontGap * gapDiff;
        block->score -= block->targetBackGap * gapDiff;
        
        queueAdd(queue, block);
        
        return NULL;
    }
    
    int swapped = 0;
    if (rows < cols) {
        SWAP(rowChain, colChain);
        SWAP(rows, cols);
        SWAP(block->targetFrontGap, block->queryFrontGap);
        SWAP(block->targetBackGap, block->queryBackGap);
        swapped = 1;
    }

    int row = rows / 2;

    // inclusive
    int pLeft = p + rows - cols + 1; 
    int pRight = p + 1;
    
    Chain* uRow = chainCreateView(rowChain, 0, row, 0);
    Chain* dRow = chainCreateView(rowChain, row + 1, rows - 1, 1);
    
    Chain* uCol = chainCreateView(colChain, 0, cols - 1, 0);
    Chain* dCol = chainCreateView(colChain, 0, cols - 1, 1);
    
    int* uScr;
    int* uAff;
    int* dScr;
    int* dAff;

    if (cardsLen == 1 || rows / 2 < MIN_DUAL_LEN || cols < MIN_DUAL_LEN) {
    
        nwLinearDataGpu(&uScr, &uAff, uRow, block->queryFrontGap, uCol, 
            block->targetFrontGap, scorer, pLeft, pRight, cards[0], NULL);
            
        nwLinearDataGpu(&dScr, &dAff, dRow, block->queryBackGap, dCol, 
            block->targetBackGap, scorer, pLeft, pRight, cards[0], NULL);
        
    } else {
        
        Thread thread;

        nwLinearDataGpu(&uScr, &uAff, uRow, block->queryFrontGap, uCol, 
            block->targetFrontGap, scorer, pLeft, pRight, cards[1], &thread);
            
        nwLinearDataGpu(&dScr, &dAff, dRow, block->queryBackGap, dCol, 
            block->targetBackGap, scorer, pLeft, pRight, cards[0], NULL);
        
        threadJoin(thread);
    }
    
    chainDelete(uRow);
    chainDelete(dRow);
    chainDelete(uCol);
    chainDelete(dCol);
    
    int uEmpty = -gapOpen - row * gapExtend + block->queryFrontGap * gapDiff;
    int dEmpty = -gapOpen - (rows - row - 2) * gapExtend + block->queryBackGap * gapDiff;
    
    int maxScr = INT_MIN;
    int gap = 0; // boolean
    int col = -1;

    int uMaxScore = 0;
    int dMaxScore = 0;
    
    int up, down;
    for(up = -1, down = cols - 1; up < cols; ++up, --down) {
    
        int uScore = up == -1 ? uEmpty : uScr[up];
        int uAffine = up == -1 ? uEmpty : uAff[up];
        
        int dScore = down == -1 ? dEmpty : dScr[down];
        int dAffine = down == -1 ? dEmpty : dAff[down];
        
        int scr = uScore + dScore;
        int aff = uAffine + dAffine + gapDiff;
        
        int isScrAff = (uScore == uAffine) && (dScore == dAffine);
        
        if (scr > maxScr || (scr == maxScr && !isScrAff)) {
            maxScr = scr;
            gap = 0;
            col = up;
            uMaxScore = uScore;
            dMaxScore = dScore;  
        }

        if (aff >= maxScr) {
            maxScr = aff;
            gap = 1;
            col = up;   
            uMaxScore = uAffine + gapDiff;
            dMaxScore = dAffine + gapDiff;
        }
    }
    
    free(uScr);
    free(uAff);
    free(dScr);
    free(dAff);

    if (block->score != NO_SCORE) {
        ASSERT(maxScr == block->score, "score: %d, found: %d", block->score, maxScr);
    }
    
    if (swapped) {
        SWAP(rowChain, colChain);
        SWAP(rows, cols);
        SWAP(row, col);
        SWAP(block->targetFrontGap, block->queryFrontGap);
        SWAP(block->targetBackGap, block->queryBackGap);
    }
    
    uRow = chainCreateView(rowChain, 0, row, 0);
    uCol = chainCreateView(colChain, 0, col, 0);
    dRow = chainCreateView(rowChain, row + 1, rows - 1, 0);
    dCol = chainCreateView(colChain, col + 1, cols - 1, 0);
    
    Block* upBlock = (Block*) malloc(sizeof(Block));
    upBlock->score = uMaxScore;
    upBlock->queryFrontGap = block->queryFrontGap;
    upBlock->queryBackGap = swapped ? 0 : gap;
    upBlock->targetFrontGap = block->targetFrontGap;
    upBlock->targetBackGap = swapped ? gap : 0;
    upBlock->startRow = block->startRow;
    upBlock->startCol = block->startCol;
    upBlock->endRow = block->startRow + row;
    upBlock->endCol = block->startCol + col;
    
    Block* downBlock = (Block*) malloc(sizeof(Block));
    downBlock->score = dMaxScore;
    downBlock->queryFrontGap = swapped ? 0 : gap;
    downBlock->queryBackGap = block->queryBackGap;
    downBlock->targetFrontGap = swapped ? gap : 0;
    downBlock->targetBackGap = block->targetBackGap;
    downBlock->startRow = block->startRow + row + 1;
    downBlock->startCol = block->startCol + col + 1;
    downBlock->endRow = block->startRow + rows - 1;
    downBlock->endCol = block->startCol + cols - 1;
       
    HirschbergContext* upContext = (HirschbergContext*) malloc(sizeof(HirschbergContext));
    upContext->block = upBlock;
    upContext->queue = queue;
    upContext->rowChain = uRow;
    upContext->colChain = uCol;
    upContext->scorer = scorer; 
    
    HirschbergContext* downContext = (HirschbergContext*) malloc(sizeof(HirschbergContext));
    downContext->block = downBlock;
    downContext->queue = queue;
    downContext->rowChain = dRow;
    downContext->colChain = dCol;
    downContext->scorer = scorer; 
    
    if (cardsLen >= 2) {
    
        int half = cardsLen / 2;
        
        upContext->cards = cards; 
        upContext->cardsLen = half; 
        
        downContext->cards = cards + half; 
        downContext->cardsLen = cardsLen - half;
         
        Thread thread;
        
        threadCreate(&thread, hirschberg, (void*) (downContext));
        hirschberg(upContext); // master thread will work !!!!
        
        threadJoin(thread);
        
    } else {
        
        upContext->cards = cards; 
        upContext->cardsLen = cardsLen; 
        
        downContext->cards = cards; 
        downContext->cardsLen = cardsLen;
       
        hirschberg(upContext);
        hirschberg(downContext); 
    }
    
    free(upContext);
    free(downContext);
    free(block);
    
    chainDelete(uRow);
    chainDelete(uCol);
    chainDelete(dRow);
    chainDelete(dCol);
    
    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// QUEUE

static void queueAdd(Queue* queue, Block* block) {

    semaphoreWait(&(queue->mutex));
    queue->data[queue->length++] = block;
    semaphorePost(&(queue->mutex));
    
    semaphorePost(&(queue->write));
}

static void queueCreate(Queue* queue, int size) {

    semaphoreCreate(&(queue->mutex), 1);
    semaphoreCreate(&(queue->write), 0);
    semaphoreCreate(&(queue->done), 0);
    
    queue->data = (Block**) malloc(size * sizeof(Block*));
    
    queue->current = 0;
    queue->length = 0;
}

static void queueDelete(Queue* queue) {

    semaphoreDelete(&(queue->mutex));
    semaphoreDelete(&(queue->write));
    semaphoreDelete(&(queue->done));
    
    int i;
    for (i = 0; i < queue->length; ++i) {
        free(queue->data[i]->path);
        free(queue->data[i]);
    }
    
    free(queue->data);
}

static void* queueWorker(void* params) {
    
    QueueContext* contexts = (QueueContext*) params;
    
    Queue* queue = contexts->queue;
    Chain* query = contexts->query;
    Chain* target = contexts->target;
    Scorer* scorer = contexts->scorer;
    
    while (1) {
    
        semaphoreWait(&(queue->write));
        
        semaphoreWait(&(queue->mutex));
        int current = queue->current++;
        semaphorePost(&(queue->mutex));
        
        Block* block = queue->data[current];

        Chain* subRow = chainCreateView(query, block->startRow, block->endRow, 0);
        Chain* subCol = chainCreateView(target, block->startCol, block->endCol, 0);

        nwReconstructCpu(&(block->path), &(block->pathLen), &(block->outScore),
            subRow, block->queryFrontGap, block->queryBackGap, 
            subCol, block->targetFrontGap, block->targetBackGap, 
            scorer, block->score);
    
        chainDelete(subRow);
        chainDelete(subCol);
        
        semaphorePost(&(queue->done));
    }
    
    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// UTIL

static int blockCmp(const void* a_, const void* b_) {

    Block* a = *((Block**) a_);
    Block* b = *((Block**) b_);
    
    int cmp1 = a->startRow - b->startRow ;
    int cmp2 = a->startCol - b->startCol;

    return cmp1 == 0 ? cmp2 : cmp1;
}
//------------------------------------------------------------------------------
//******************************************************************************
