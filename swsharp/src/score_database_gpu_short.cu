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

#include "score_database_gpu_short.h"

#define THREADS   128
#define BLOCKS    120

#define INT4_ZERO make_int4(0, 0, 0, 0)
#define INT4_SCORE_MIN make_int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

struct ShortDatabase {
    int length;
    int* order;
    int* positions;
    int blocks;
    int* offsets;
    int* lengths;
    int* lengthsPadded;
    size_t lengthsSize;
    char4* sequences;
    int sequencesLen;
    int sequencesRows;
    int sequencesCols;
    size_t sequencesSize;
};

typedef struct ShortDatabaseGpu {
    int* scores;
    int* lengths;
    int* lengthsPadded;
    cudaArray* sequences;
    int2* hBus;
} ShortDatabaseGpu;

typedef struct Context {
    int** scores; 
    int type;
    Chain** queries;
    int queriesLen;
    ShortDatabase* shortDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int* cards;
    int cardsLen;
} Context;

typedef struct KernelContext {
    int* scores; 
    int type;
    Chain** queries;
    int queriesLen;
    ShortDatabase* shortDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int card;
    int queriesStart;
    int queriesStep;
    int blocksStart;
    int blocksStep;
} KernelContext;

static __constant__ int gapOpen_;
static __constant__ int gapExtend_;

static __constant__ int rows_;
static __constant__ int rowsPadded_;
static __constant__ int width_;

texture<int, 2, cudaReadModeElementType> seqsTexture;
texture<char4, 2, cudaReadModeElementType> subTexture;

//******************************************************************************
// PUBLIC

extern ShortDatabase* shortDatabaseCreate(Chain** database, int databaseLen);

extern void shortDatabaseDelete(ShortDatabase* shortDatabase);

extern void scoreShortDatabaseGpu(int** scores, int type, Chain* query, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread);

extern void scoreShortDatabasesGpu(int** scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void scoreDatabase(int** scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

static void* scoreDatabaseThread(void* param);

// cpu kernels
static void* kernel(void* param);

static void kernelSingle(int* scores, int type, Chain* query,
    ShortDatabase* shortDatabase, ShortDatabaseGpu* shortDatabaseGpu, 
    Scorer* scorer, int blocksStart, int blocksStep);

// gpu database preparation
static ShortDatabaseGpu* shortDatabaseGpuCreate(ShortDatabase* shortDatabase,
    int* indexes, int indexesLen);

static void shortDatabaseGpuDelete(ShortDatabaseGpu* shortDatabaseGpu);

// gpu kernels
__device__ static int gap(int index);

__global__ static void hwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int off);

__global__ static void nwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int off);

__global__ static void swSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int off);

// utils
static int* createOrderArray(Chain** database, int databaseLen);

static int int2CmpY(const void* a_, const void* b_);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern ShortDatabase* shortDatabaseCreate(Chain** database, int databaseLen) {
    
    int* order = createOrderArray(database, databaseLen);
    
    int* positions = (int*) malloc(databaseLen * sizeof(int));
    
    for (int i = 0; i < databaseLen; ++i) {
        positions[order[i]] = i;
    }
    
    // create gpu database
    int sequencesCols = THREADS * BLOCKS;
    int sequencesRows = 0;

    int blocks = 0;
    
    // calculate sequence grid
    for (int i = sequencesCols - 1; i < databaseLen; i += sequencesCols) {
        int n = chainGetLength(database[order[i]]);
        sequencesRows += (n + (4 - n % 4) % 4) / 4;
        blocks++;
    }
    
    if (databaseLen % sequencesCols != 0) {
        int n = chainGetLength(database[order[databaseLen - 1]]);
        sequencesRows += (n + (4 - n % 4) % 4) / 4;
        blocks++;
    }
    
    // initialize structures
    int* offsets = (int*) malloc(blocks * sizeof(int));
    offsets[0] = 0;
    
    size_t lengthsSize = blocks * sequencesCols * sizeof(int);
    int* lengths = (int*) malloc(lengthsSize);
    memset(lengths, 0, lengthsSize);
    
    int* lengthsPadded = (int*) malloc(lengthsSize);
    memset(lengthsPadded, 0, lengthsSize);
    
    size_t sequencesSize = sequencesRows * sequencesCols * sizeof(char4);
    char4* sequences = (char4*) malloc(sequencesSize);
    
    // tmp
    size_t sequenceSize = chainGetLength(database[order[databaseLen - 1]]) + 4;
    char* sequence = (char*) malloc(sequenceSize);

    for(int i = 0, j = 0, cx = 0, cy = 0; i < databaseLen; i++){

        //get the sequence and its length
        Chain* chain = database[order[i]];
        int n = chainGetLength(chain);    
        
        lengths[j * sequencesCols + cx] = n;
        
        chainCopyCodes(chain, sequence);
        memset(sequence + n, 255, 4 * sizeof(char));

        n = n + (4 - n % 4) % 4;
        int n4 = n / 4;

        lengthsPadded[j * sequencesCols + cx] = n4;
        
        char4* ptr = sequences + cy * sequencesCols + cx;
        for(int k = 0; k < n; k += 4){
            ptr->x = sequence[k];
            ptr->y = sequence[k + 1];
            ptr->z = sequence[k + 2];
            ptr->w = sequence[k + 3];
            ptr += sequencesCols;
        }

        cx++;
        
        if(cx == sequencesCols){
            offsets[j + 1] = offsets[j] + n4;
            cx = 0;
            cy += n4;
            j++;
        }
    }
    
    free(sequence);
    
    ShortDatabase* shortDatabase = 
        (ShortDatabase*) malloc(sizeof(struct ShortDatabase));
    
    shortDatabase->length = databaseLen;
    shortDatabase->order = order;
    shortDatabase->positions = positions;
    shortDatabase->blocks = blocks;
    shortDatabase->offsets = offsets;
    shortDatabase->lengths = lengths;
    shortDatabase->lengthsPadded = lengthsPadded;
    shortDatabase->lengthsSize = lengthsSize;
    shortDatabase->sequences = sequences;
    shortDatabase->sequencesSize = sequencesSize;
    shortDatabase->sequencesLen = databaseLen;
    shortDatabase->sequencesRows = sequencesRows;
    shortDatabase->sequencesCols = sequencesCols;
    
    return shortDatabase;
}
    
extern void shortDatabaseDelete(ShortDatabase* shortDatabase) {

    free(shortDatabase->order);
    free(shortDatabase->positions);
    free(shortDatabase->offsets);
    free(shortDatabase->lengths);
    free(shortDatabase->lengthsPadded);
    free(shortDatabase->sequences);
    
    free(shortDatabase);
    shortDatabase = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// DATABASE SCORING

extern void scoreShortDatabaseGpu(int** scores, int type, Chain* query, 
    ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, &query, 1, shortDatabase, scorer, indexes, 
        indexesLen, cards, cardsLen, thread);
}

extern void scoreShortDatabasesGpu(int** scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, queries, queriesLen, shortDatabase, scorer,
        indexes, indexesLen, cards, cardsLen, thread);
}

//------------------------------------------------------------------------------

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// DATABASE SCORING

static void scoreDatabase(int** scores, int type, Chain** queries, 
    int queriesLen, ShortDatabase* shortDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    
    ASSERT(cardsLen > 0, "no GPUs available");
    
    Context* param = (Context*) malloc(sizeof(Context));
    
    param->scores = scores;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->shortDatabase = shortDatabase;
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
    ShortDatabase* shortDatabase = context->shortDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int databaseLen = shortDatabase->length;
    int* order = shortDatabase->order;
    
    //**************************************************************************
    // SOLVE MULTICARDED
    
    // there is no level of paralelization available
    if (queriesLen == 1 && shortDatabase->blocks == 1) {
        cardsLen = 1;
    }
    
    int threadNmr = cardsLen;
    
    Thread* threads = (Thread*) malloc((threadNmr - 1) * sizeof(Thread));
    
    int* unordered = (int*) malloc(queriesLen * databaseLen * sizeof(int));
    
    KernelContext* contexts = 
        (KernelContext*) malloc(threadNmr * sizeof(KernelContext));

    for (int i = 0; i < threadNmr; ++i) {
    
        contexts[i].scores = unordered;
        contexts[i].type = type;
        contexts[i].queries = queries;
        contexts[i].queriesLen = queriesLen;
        contexts[i].shortDatabase = shortDatabase;
        contexts[i].scorer = scorer;
        contexts[i].card = cards[i];
        contexts[i].indexes = indexes;
        contexts[i].indexesLen = indexesLen;
        
        if (threadNmr < queriesLen) {
            // one query, single card
            contexts[i].queriesStart = i;
            contexts[i].queriesStep = cardsLen;
            contexts[i].blocksStart = 0;
            contexts[i].blocksStep = 1;
        } else {
            // one query, multiple cards
            contexts[i].queriesStart = 0;
            contexts[i].queriesStep = 1;
            contexts[i].blocksStart = i;
            contexts[i].blocksStep = cardsLen;
        }
    }

    for (int i = 0; i < threadNmr - 1; ++i) {    
        threadCreate(&threads[i], kernel, &contexts[i]);
    }
    
    kernel(&contexts[threadNmr - 1]);
    
    for (int i = 0; i < threadNmr - 1; ++i) {
        threadJoin(threads[i]);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // SAVE SCORES
    
    *scores = (int*) malloc(queriesLen * databaseLen * sizeof(int));
    
    // copy
    for (int i = 0; i < queriesLen; ++i) {
        for (int j = 0; j < databaseLen; ++j) {
            (*scores)[i * databaseLen + order[j]] = unordered[i * databaseLen + j];
        }
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    free(unordered);
    free(threads);
    free(contexts);

    free(param);
    
    //**************************************************************************
    
    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void* kernel(void* param) {

    KernelContext* context = (KernelContext*) param;
    
    int* scores = context->scores;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    ShortDatabase* shortDatabase = context->shortDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int card = context->card;
    int queriesStart = context->queriesStart;
    int queriesStep = context->queriesStep;
    int blocksStart = context->blocksStart;
    int blocksStep = context->blocksStep;
    
    // set card
    int currentCard;
    CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
    if (currentCard != card) {
        CUDA_SAFE_CALL(cudaThreadExit());
        CUDA_SAFE_CALL(cudaSetDevice(card));
    }
    
    // prepare gpu db
    ShortDatabaseGpu* shortDatabaseGpu = shortDatabaseGpuCreate(shortDatabase, 
        indexes, indexesLen);

    // solve
    for (int i = queriesStart; i < queriesLen; i += queriesStep) {
    
        Chain* query = queries[i];
        int offset = i * shortDatabase->length;
        
        kernelSingle(scores + offset, type, query, shortDatabase, 
            shortDatabaseGpu, scorer, blocksStart, blocksStep);
    }
    
    shortDatabaseGpuDelete(shortDatabaseGpu);

    return NULL;
}

static void kernelSingle(int* scores, int type, Chain* query,
    ShortDatabase* shortDatabase, ShortDatabaseGpu* shortDatabaseGpu, 
    Scorer* scorer, int blocksStart, int blocksStep) {
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    
    int* offsets = shortDatabase->offsets;
    int blocks = shortDatabase->blocks;
    
    int sequencesCols = shortDatabase->sequencesCols;
    int sequencesLen = shortDatabase->sequencesLen;
    
    int* scoresGpu = shortDatabaseGpu->scores;
    int* lengths = shortDatabaseGpu->lengths;
    int* lengthsPadded = shortDatabaseGpu->lengthsPadded;
    
    int2* hBus = shortDatabaseGpu->hBus;
    
    //**************************************************************************
    // CREATE QUERY PROFILE
    
    int rows = chainGetLength(query);
    int rowsGpu = rows + (8 - rows % 8) % 8;
    
    size_t rowSize = rows * sizeof(char);
    char* row = (char*) malloc(rowSize);
    chainCopyCodes(query, row);

    int subLen = scorerGetMaxCode(scorer) + 1;
    size_t subSize = rowsGpu * subLen * sizeof(char);
    char4* subCpu = (char4*) malloc(subSize);
    memset(subCpu, 0, subSize);
    for (int i = 0; i < rowsGpu / 4; ++i) {
        for (int j = 0; j < subLen - 1; ++j) {
            char4 scr;
            scr.x = i * 4 + 0 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 0], j);
            scr.y = i * 4 + 1 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 1], j);
            scr.z = i * 4 + 2 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 2], j);
            scr.w = i * 4 + 3 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 3], j);
            subCpu[i * subLen + j] = scr;
        }
    }
    
    cudaArray* subArray;
    int subH = rowsGpu / 4;
    CUDA_SAFE_CALL(cudaMallocArray(&subArray, &subTexture.channelDesc, subLen, subH)); 
    CUDA_SAFE_CALL(cudaMemcpyToArray (subArray, 0, 0, subCpu, subSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTextureToArray(subTexture, subArray));
    subTexture.addressMode[0] = cudaAddressModeClamp;
    subTexture.addressMode[1] = cudaAddressModeClamp;
    subTexture.filterMode = cudaFilterModePoint;
    subTexture.normalized = false;

    //**************************************************************************
    
    //**************************************************************************
    // INIT GPU
    
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rows_, &rows, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rowsPadded_, &rowsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapOpen_, &gapOpen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapExtend_, &gapExtend, sizeof(int)));
    
    //**************************************************************************
    
    //**************************************************************************
    // SOVLE
    
    void (*function)(int*, int2*, int*, int*, int);
    switch (type) {
    case SW_ALIGN: 
        function = swSolveShortGpu;
        break;
    case NW_ALIGN: 
        function = nwSolveShortGpu;
        break;
    case HW_ALIGN:
        function = hwSolveShortGpu;
        break;
    default:
        ERROR("Wrong align type");
    }
    
    for (int i = blocksStart; i < blocks; i += blocksStep) {

        int colOff = i * sequencesCols;
        int rowOff = offsets[i];
        
        function<<<BLOCKS, THREADS>>>(scoresGpu, hBus, lengths + colOff, 
            lengthsPadded + colOff, rowOff);

        // copy scores from the GPU
        size_t size = min(sequencesCols, sequencesLen - colOff) * sizeof(int);
        CUDA_SAFE_CALL(cudaMemcpy(scores + colOff, scoresGpu, size, FROM_GPU));
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY
    
    free(subCpu);
    free(row);
    
    CUDA_SAFE_CALL(cudaFreeArray(subArray));
    
    //**************************************************************************
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU DATABASE PREPARE

extern ShortDatabaseGpu* shortDatabaseGpuCreate(ShortDatabase* shortDatabase,
    int* indexes, int indexesLen) {
    
    int* lengthsPadded;
    
    // filter and copy padded lengths
    if (indexes == NULL) {
        lengthsPadded = shortDatabase->lengthsPadded;
    } else {
        lengthsPadded = (int*) malloc(shortDatabase->lengthsSize);
        memset(lengthsPadded, 0, shortDatabase->lengthsSize);
        
        int length = shortDatabase->length;
        
        for (int i = 0; i < indexesLen; ++i) {
        
            int index = indexes[i];
            ASSERT(index < length, "wrong index: %d\n", index);
            
            int ord = shortDatabase->positions[index];
            lengthsPadded[ord] = shortDatabase->lengthsPadded[ord];
        }
    }
    
    size_t lengthsSize = shortDatabase->lengthsSize;
    int* lengthsPaddedGpu;
    CUDA_SAFE_CALL(cudaMalloc(&lengthsPaddedGpu, lengthsSize));
    CUDA_SAFE_CALL(cudaMemcpy(lengthsPaddedGpu, lengthsPadded, lengthsSize, TO_GPU));
    
    if (indexes != NULL) {
        free(lengthsPadded);
    }

    // copy lengths
    int* lengths = shortDatabase->lengths;
    int* lengthsGpu;
    CUDA_SAFE_CALL(cudaMalloc(&lengthsGpu, lengthsSize));
    CUDA_SAFE_CALL(cudaMemcpy(lengthsGpu, lengths, lengthsSize, TO_GPU));
    
    // copy sequences
    int sequencesCols = shortDatabase->sequencesCols;
    int sequencesRows = shortDatabase->sequencesRows;
    size_t sequencesSize = shortDatabase->sequencesSize;
    char4* sequences = shortDatabase->sequences;
    cudaArray* sequencesGpu;
    cudaChannelFormatDesc channel = seqsTexture.channelDesc;
    CUDA_SAFE_CALL(cudaMallocArray(&sequencesGpu, &channel, sequencesCols, sequencesRows)); 
    CUDA_SAFE_CALL(cudaMemcpyToArray(sequencesGpu, 0, 0, sequences, sequencesSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTextureToArray(seqsTexture, sequencesGpu));
    
    // init scores
    int* scoresGpu;
    size_t scoresSize = sequencesCols * sizeof(int);
    CUDA_SAFE_CALL(cudaMalloc(&scoresGpu, scoresSize));
    
    // init h bus
    int* offsets = shortDatabase->offsets;
    int blocks = shortDatabase->blocks;
    int2* hBusGpu;
    int hBusHeight = (sequencesRows - offsets[blocks - 1]) * 4;
    size_t hBusSize = sequencesCols * hBusHeight * sizeof(int2);
    CUDA_SAFE_CALL(cudaMalloc(&hBusGpu, hBusSize));
    
    // constants
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(width_, &sequencesCols, sizeof(int)));
    
    ShortDatabaseGpu* shortDatabaseGpu = 
        (ShortDatabaseGpu*) malloc(sizeof(struct ShortDatabaseGpu));
    
    shortDatabaseGpu->scores = scoresGpu;
    shortDatabaseGpu->lengths = lengthsGpu;
    shortDatabaseGpu->lengthsPadded = lengthsPaddedGpu;
    shortDatabaseGpu->sequences = sequencesGpu;
    shortDatabaseGpu->hBus = hBusGpu;

    return shortDatabaseGpu;
}

extern void shortDatabaseGpuDelete(ShortDatabaseGpu* shortDatabaseGpu) {

    CUDA_SAFE_CALL(cudaFree(shortDatabaseGpu->scores));
    CUDA_SAFE_CALL(cudaFree(shortDatabaseGpu->lengths));
    CUDA_SAFE_CALL(cudaFree(shortDatabaseGpu->lengthsPadded));
    CUDA_SAFE_CALL(cudaFreeArray(shortDatabaseGpu->sequences));
    CUDA_SAFE_CALL(cudaFree(shortDatabaseGpu->hBus));
    
    CUDA_SAFE_CALL(cudaUnbindTexture(seqsTexture));
    
    free(shortDatabaseGpu);
    shortDatabaseGpu = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

__device__ static int gap(int index) {
    return (-gapOpen_ - index * gapExtend_) * (index >= 0);
}

__global__ static void hwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int off) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int cols = lengthsPadded[id];

    if (cols == 0) {
        scores[id] = NO_SCORE;
        return;
    }
    
    int score = SCORE_MIN;
    
    int4 scrUp;
    int4 affUp;
    int4 mchUp;
    
    int4 scrDown;
    int4 affDown;
    int4 mchDown;
    
    int2 wBus;
    int del;
    
    int lastRow = rows_ - 1;
    int realCols = lengths[id];
    
    for (int j = 0; j < cols * 4; ++j) {
        hBus[j * width_ + id] = make_int2(0, SCORE_MIN);
    }
    
    for (int i = 0; i < rowsPadded_; i += 8) {
    
        scrUp = make_int4(gap(i), gap(i + 1), gap(i + 2), gap(i + 3));
        affUp = INT4_SCORE_MIN;
        mchUp = make_int4(gap(i - 1), gap(i), gap(i + 1), gap(i + 2));
        
        scrDown = make_int4(gap(i + 4), gap(i + 5), gap(i + 6), gap(i + 7));
        affDown = INT4_SCORE_MIN;
        mchDown = make_int4(gap(i + 3), gap(i + 4), gap(i + 5), gap(i + 6));
        
        for (int j = 0; j < cols; ++j) {
        
            int columnCodes = tex2D(seqsTexture, id, j + off);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
            
                int validCol = (j * 4 + k) < realCols;
                
                wBus = hBus[(j * 4 + k) * width_ + id];
                
                char code = (columnCodes >> (k << 3));
                char4 rowScores = tex2D(subTexture, code, i / 4);
                
                del = max(wBus.x - gapOpen_, wBus.y - gapExtend_);
                affUp.x = max(scrUp.x - gapOpen_, affUp.x - gapExtend_);
                scrUp.x = mchUp.x + rowScores.x; 
                scrUp.x = max(scrUp.x, del);
                scrUp.x = max(scrUp.x, affUp.x);
                mchUp.x = wBus.x;
                if (i + 0 == lastRow && validCol) score = max(score, scrUp.x);
                
                del = max(scrUp.x - gapOpen_, del - gapExtend_);
                affUp.y = max(scrUp.y - gapOpen_, affUp.y - gapExtend_);
                scrUp.y = mchUp.y + rowScores.y; 
                scrUp.y = max(scrUp.y, del);
                scrUp.y = max(scrUp.y, affUp.y);
                mchUp.y = scrUp.x;
                if (i + 1 == lastRow && validCol) score = max(score, scrUp.y);
                
                del = max(scrUp.y - gapOpen_, del - gapExtend_);
                affUp.z = max(scrUp.z - gapOpen_, affUp.z - gapExtend_);
                scrUp.z = mchUp.z + rowScores.z; 
                scrUp.z = max(scrUp.z, del);
                scrUp.z = max(scrUp.z, affUp.z);
                mchUp.z = scrUp.y;
                if (i + 2 == lastRow && validCol) score = max(score, scrUp.z);
                
                del = max(scrUp.z - gapOpen_, del - gapExtend_);
                affUp.w = max(scrUp.w - gapOpen_, affUp.w - gapExtend_);
                scrUp.w = mchUp.w + rowScores.w; 
                scrUp.w = max(scrUp.w, del);
                scrUp.w = max(scrUp.w, affUp.w);
                mchUp.w = scrUp.z;
                if (i + 3 == lastRow && validCol) score = max(score, scrUp.w);

                rowScores = tex2D(subTexture, code, i / 4 + 1);
                
                del = max(scrUp.w - gapOpen_, del - gapExtend_);
                affDown.x = max(scrDown.x - gapOpen_, affDown.x - gapExtend_);
                scrDown.x = mchDown.x + rowScores.x; 
                scrDown.x = max(scrDown.x, del);
                scrDown.x = max(scrDown.x, affDown.x);
                mchDown.x = scrUp.w;
                if (i + 4 == lastRow && validCol) score = max(score, scrDown.x);
                
                del = max(scrDown.x - gapOpen_, del - gapExtend_);
                affDown.y = max(scrDown.y - gapOpen_, affDown.y - gapExtend_);
                scrDown.y = mchDown.y + rowScores.y; 
                scrDown.y = max(scrDown.y, del);
                scrDown.y = max(scrDown.y, affDown.y);
                mchDown.y = scrDown.x;
                if (i + 5 == lastRow && validCol) score = max(score, scrDown.y);
                
                del = max(scrDown.y - gapOpen_, del - gapExtend_);
                affDown.z = max(scrDown.z - gapOpen_, affDown.z - gapExtend_);
                scrDown.z = mchDown.z + rowScores.z; 
                scrDown.z = max(scrDown.z, del);
                scrDown.z = max(scrDown.z, affDown.z);
                mchDown.z = scrDown.y;
                if (i + 6 == lastRow && validCol) score = max(score, scrDown.z);
                
                del = max(scrDown.z - gapOpen_, del - gapExtend_);
                affDown.w = max(scrDown.w - gapOpen_, affDown.w - gapExtend_);
                scrDown.w = mchDown.w + rowScores.w; 
                scrDown.w = max(scrDown.w, del);
                scrDown.w = max(scrDown.w, affDown.w);
                mchDown.w = scrDown.z;
                if (i + 7 == lastRow && validCol) score = max(score, scrDown.w);
                
                wBus.x = scrDown.w;
                wBus.y = del;
                
                hBus[(j * 4 + k) * width_ + id] = wBus;
            }
        }
    }
    
    scores[id] = score;
}

__global__ static void nwSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int off) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int cols = lengthsPadded[id];

    if (cols == 0) {
        scores[id] = NO_SCORE;
        return;
    }
    
    int score = SCORE_MIN;
    
    int4 scrUp;
    int4 affUp;
    int4 mchUp;
    
    int4 scrDown;
    int4 affDown;
    int4 mchDown;
    
    int2 wBus;
    int del;
    
    int lastRow = rows_ - 1;
    int realCols = lengths[id];
    
    for (int j = 0; j < cols * 4; ++j) {
        hBus[j * width_ + id] = make_int2(gap(j), SCORE_MIN);
    }
    
    for (int i = 0; i < rowsPadded_; i += 8) {
    
        scrUp = make_int4(gap(i), gap(i + 1), gap(i + 2), gap(i + 3));
        affUp = INT4_SCORE_MIN;
        mchUp = make_int4(gap(i - 1), gap(i), gap(i + 1), gap(i + 2));
        
        scrDown = make_int4(gap(i + 4), gap(i + 5), gap(i + 6), gap(i + 7));
        affDown = INT4_SCORE_MIN;
        mchDown = make_int4(gap(i + 3), gap(i + 4), gap(i + 5), gap(i + 6));
        
        for (int j = 0; j < cols; ++j) {
        
            int columnCodes = tex2D(seqsTexture, id, j + off);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
            
                int lastCol = (j * 4 + k) == (realCols - 1);
                
                wBus = hBus[(j * 4 + k) * width_ + id];
                
                char code = (columnCodes >> (k << 3));
                char4 rowScores = tex2D(subTexture, code, i / 4);
                
                del = max(wBus.x - gapOpen_, wBus.y - gapExtend_);
                affUp.x = max(scrUp.x - gapOpen_, affUp.x - gapExtend_);
                scrUp.x = mchUp.x + rowScores.x; 
                scrUp.x = max(scrUp.x, del);
                scrUp.x = max(scrUp.x, affUp.x);
                mchUp.x = wBus.x;
                if (i + 0 == lastRow && lastCol) score = scrUp.x;
                
                del = max(scrUp.x - gapOpen_, del - gapExtend_);
                affUp.y = max(scrUp.y - gapOpen_, affUp.y - gapExtend_);
                scrUp.y = mchUp.y + rowScores.y; 
                scrUp.y = max(scrUp.y, del);
                scrUp.y = max(scrUp.y, affUp.y);
                mchUp.y = scrUp.x;
                if (i + 1 == lastRow && lastCol) score = scrUp.y;
                
                del = max(scrUp.y - gapOpen_, del - gapExtend_);
                affUp.z = max(scrUp.z - gapOpen_, affUp.z - gapExtend_);
                scrUp.z = mchUp.z + rowScores.z; 
                scrUp.z = max(scrUp.z, del);
                scrUp.z = max(scrUp.z, affUp.z);
                mchUp.z = scrUp.y;
                if (i + 2 == lastRow && lastCol) score = scrUp.z;
                
                del = max(scrUp.z - gapOpen_, del - gapExtend_);
                affUp.w = max(scrUp.w - gapOpen_, affUp.w - gapExtend_);
                scrUp.w = mchUp.w + rowScores.w; 
                scrUp.w = max(scrUp.w, del);
                scrUp.w = max(scrUp.w, affUp.w);
                mchUp.w = scrUp.z;
                if (i + 3 == lastRow && lastCol) score = scrUp.w;

                rowScores = tex2D(subTexture, code, i / 4 + 1);
                
                del = max(scrUp.w - gapOpen_, del - gapExtend_);
                affDown.x = max(scrDown.x - gapOpen_, affDown.x - gapExtend_);
                scrDown.x = mchDown.x + rowScores.x; 
                scrDown.x = max(scrDown.x, del);
                scrDown.x = max(scrDown.x, affDown.x);
                mchDown.x = scrUp.w;
                if (i + 4 == lastRow && lastCol) score = scrDown.x;
                
                del = max(scrDown.x - gapOpen_, del - gapExtend_);
                affDown.y = max(scrDown.y - gapOpen_, affDown.y - gapExtend_);
                scrDown.y = mchDown.y + rowScores.y; 
                scrDown.y = max(scrDown.y, del);
                scrDown.y = max(scrDown.y, affDown.y);
                mchDown.y = scrDown.x;
                if (i + 5 == lastRow && lastCol) score = scrDown.y;
                
                del = max(scrDown.y - gapOpen_, del - gapExtend_);
                affDown.z = max(scrDown.z - gapOpen_, affDown.z - gapExtend_);
                scrDown.z = mchDown.z + rowScores.z; 
                scrDown.z = max(scrDown.z, del);
                scrDown.z = max(scrDown.z, affDown.z);
                mchDown.z = scrDown.y;
                if (i + 6 == lastRow && lastCol) score = scrDown.z;
                
                del = max(scrDown.z - gapOpen_, del - gapExtend_);
                affDown.w = max(scrDown.w - gapOpen_, affDown.w - gapExtend_);
                scrDown.w = mchDown.w + rowScores.w; 
                scrDown.w = max(scrDown.w, del);
                scrDown.w = max(scrDown.w, affDown.w);
                mchDown.w = scrDown.z;
                if (i + 7 == lastRow && lastCol) score = scrDown.w;
                
                wBus.x = scrDown.w;
                wBus.y = del;
                
                hBus[(j * 4 + k) * width_ + id] = wBus;
            }
        }
    }
    
    scores[id] = score;
}

__global__ static void swSolveShortGpu(int* scores, int2* hBus, int* lengths, 
    int* lengthsPadded, int off) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int cols = lengthsPadded[id];

    if (cols == 0) {
        scores[id] = NO_SCORE;
        return;
    }
    
    int score = 0;
    
    int4 scrUp;
    int4 affUp;
    int4 mchUp;
    
    int4 scrDown;
    int4 affDown;
    int4 mchDown;
    
    int2 wBus;
    int del;
    
    for (int j = 0; j < cols * 4; ++j) {
        hBus[j * width_ + id] = make_int2(0, SCORE_MIN);
    }
    
    for (int i = 0; i < rowsPadded_; i += 8) {
    
        scrUp = INT4_ZERO;
        affUp = INT4_SCORE_MIN;
        mchUp = INT4_ZERO;
        
        scrDown = INT4_ZERO;
        affDown = INT4_SCORE_MIN;
        mchDown = INT4_ZERO;
        
        for (int j = 0; j < cols; ++j) {
        
            int columnCodes = tex2D(seqsTexture, id, j + off);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
            
                wBus = hBus[(j * 4 + k) * width_ + id];
                
                char code = (columnCodes >> (k << 3));
                char4 rowScores = tex2D(subTexture, code, i / 4);
                
                del = max(wBus.x - gapOpen_, wBus.y - gapExtend_);
                affUp.x = max(scrUp.x - gapOpen_, affUp.x - gapExtend_);
                scrUp.x = mchUp.x + rowScores.x; 
                scrUp.x = max(scrUp.x, del);
                scrUp.x = max(scrUp.x, affUp.x);
                scrUp.x = max(scrUp.x, 0);
                mchUp.x = wBus.x;
                score = max(score, scrUp.x);
                
                del = max(scrUp.x - gapOpen_, del - gapExtend_);
                affUp.y = max(scrUp.y - gapOpen_, affUp.y - gapExtend_);
                scrUp.y = mchUp.y + rowScores.y; 
                scrUp.y = max(scrUp.y, del);
                scrUp.y = max(scrUp.y, affUp.y);
                scrUp.y = max(scrUp.y, 0);
                mchUp.y = scrUp.x;
                score = max(score, scrUp.y);
                
                del = max(scrUp.y - gapOpen_, del - gapExtend_);
                affUp.z = max(scrUp.z - gapOpen_, affUp.z - gapExtend_);
                scrUp.z = mchUp.z + rowScores.z; 
                scrUp.z = max(scrUp.z, del);
                scrUp.z = max(scrUp.z, affUp.z);
                scrUp.z = max(scrUp.z, 0);
                mchUp.z = scrUp.y;
                score = max(score, scrUp.z);
                
                del = max(scrUp.z - gapOpen_, del - gapExtend_);
                affUp.w = max(scrUp.w - gapOpen_, affUp.w - gapExtend_);
                scrUp.w = mchUp.w + rowScores.w; 
                scrUp.w = max(scrUp.w, del);
                scrUp.w = max(scrUp.w, affUp.w);
                scrUp.w = max(scrUp.w, 0);
                mchUp.w = scrUp.z;
                score = max(score, scrUp.w);

                rowScores = tex2D(subTexture, code, i / 4 + 1);
                
                del = max(scrUp.w - gapOpen_, del - gapExtend_);
                affDown.x = max(scrDown.x - gapOpen_, affDown.x - gapExtend_);
                scrDown.x = mchDown.x + rowScores.x; 
                scrDown.x = max(scrDown.x, del);
                scrDown.x = max(scrDown.x, affDown.x);
                scrDown.x = max(scrDown.x, 0);
                mchDown.x = scrUp.w;
                score = max(score, scrDown.x);
                
                del = max(scrDown.x - gapOpen_, del - gapExtend_);
                affDown.y = max(scrDown.y - gapOpen_, affDown.y - gapExtend_);
                scrDown.y = mchDown.y + rowScores.y; 
                scrDown.y = max(scrDown.y, del);
                scrDown.y = max(scrDown.y, affDown.y);
                scrDown.y = max(scrDown.y, 0);
                mchDown.y = scrDown.x;
                score = max(score, scrDown.y);
                
                del = max(scrDown.y - gapOpen_, del - gapExtend_);
                affDown.z = max(scrDown.z - gapOpen_, affDown.z - gapExtend_);
                scrDown.z = mchDown.z + rowScores.z; 
                scrDown.z = max(scrDown.z, del);
                scrDown.z = max(scrDown.z, affDown.z);
                scrDown.z = max(scrDown.z, 0);
                mchDown.z = scrDown.y;
                score = max(score, scrDown.z);
                
                del = max(scrDown.z - gapOpen_, del - gapExtend_);
                affDown.w = max(scrDown.w - gapOpen_, affDown.w - gapExtend_);
                scrDown.w = mchDown.w + rowScores.w; 
                scrDown.w = max(scrDown.w, del);
                scrDown.w = max(scrDown.w, affDown.w);
                scrDown.w = max(scrDown.w, 0);
                mchDown.w = scrDown.z;
                score = max(score, scrDown.w);
                
                wBus.x = scrDown.w;
                wBus.y = del;
                
                hBus[(j * 4 + k) * width_ + id] = wBus;
            }
        }
    }
    
    scores[id] = score;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// UTILS

static int* createOrderArray(Chain** database, int databaseLen) {

    int2* packed = (int2*) malloc(databaseLen * sizeof(int2));
    
    for (int i = 0; i < databaseLen; ++i) {
        packed[i].x = i;
        packed[i].y = chainGetLength(database[i]);
    }
    
    qsort(packed, databaseLen, sizeof(int2), int2CmpY);

    int* order = (int*) malloc(databaseLen * sizeof(int));
    
    for (int i = 0; i < databaseLen; ++i) {
        order[i] = packed[i].x;
    }
    
    free(packed);
    
    return order;
}

static int int2CmpY(const void* a_, const void* b_) {

    int2 a = *((int2*) a_);
    int2 b = *((int2*) b_);
    
    return a.y - b.y;
}

//------------------------------------------------------------------------------
//******************************************************************************
