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

#define THREADS   64
#define BLOCKS    240

#define MAX_THREADS THREADS

#define INT4_ZERO make_int4(0, 0, 0, 0)
#define INT4_SCORE_MIN make_int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

struct LongDatabase {
    int length;
    char* codes;
    int codesLen;
    size_t codesSize;
    int* lengths;
    int* starts;
};

typedef struct LongDatabaseGpu {
    int length;
    char* codes;
    int3* data;
    int2* hBus;
    int* scores;
    size_t scoresSize;
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

typedef struct KernelContext {
    int* scores; 
    int type;
    Chain** queries;
    int queriesLen;
    LongDatabase* longDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int card;
    int queriesStart;
    int queriesStep;
} KernelContext;

typedef struct Atom {
    int mch;
    int2 up;
    int4 lScr;
    int4 lAff;
    int4 rScr;
    int4 rAff;
} Atom;

static __constant__ int gapOpen_;
static __constant__ int gapExtend_;

static __constant__ int rows_;
static __constant__ int rowsPadded_; 
static __constant__ int length_;
static __constant__ int iters_;

texture<char4, 2, cudaReadModeElementType> subTexture;

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

static void kernelSingle(int* scores, int type, Chain* query,
    LongDatabase* longDatabase, LongDatabaseGpu* longDatabaseGpu, 
    Scorer* scorer);

// gpu database preparation
static LongDatabaseGpu* longDatabaseGpuCreate(LongDatabase* longDatabase,
    int* indexes, int indexesLen);

static void longDatabaseGpuDelete(LongDatabaseGpu* longDatabaseGpu);

// gpu kernels
__global__ void hwSolve(int* scores, char* codes, int2* hBus, int3* data);

__global__ void nwSolve(int* scores, char* codes, int2* hBus, int3* data);

__global__ void swSolve(int* scores, char* codes, int2* hBus, int3* data);

__device__ static int gap(int index);

__device__ void hwSolveSingle(int* scores, char* codes, int2* hBus, int3 data);

__device__ void nwSolveSingle(int* scores, char* codes, int2* hBus, int3 data);

__device__ void swSolveSingle(int* scores, char* codes, int2* hBus, int3 data);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern LongDatabase* longDatabaseCreate(Chain** database, int databaseLen) {
    
    int* lengths = (int*) malloc(databaseLen * sizeof(int));
    int* starts = (int*) malloc(databaseLen * sizeof(int));
    
    int codesLen = 0;
    for (int i = 0; i < databaseLen; ++i) {

        int n = chainGetLength(database[i]);
        
        lengths[i] = n;
        starts[i] = codesLen;
        
        codesLen += n;        
    }
    
    size_t codesSize = codesLen * sizeof(char);
    char* codes = (char*) malloc(codesSize);
    
    for (int i = 0; i < databaseLen; ++i) {
        chainCopyCodes(database[i], codes + starts[i]);      
    }
    
    LongDatabase* longDatabase = 
        (LongDatabase*) malloc(sizeof(struct LongDatabase));
    
    longDatabase->length = databaseLen;
    longDatabase->codes = codes;
    longDatabase->codesLen = codesLen;
    longDatabase->codesSize = codesSize;
    longDatabase->lengths = lengths;
    longDatabase->starts = starts;
    
    return longDatabase;
}

extern void longDatabaseDelete(LongDatabase* longDatabase) {
    
    free(longDatabase->codes);
    free(longDatabase->starts);
    free(longDatabase->lengths);
    
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
    
    int databaseLen = longDatabase->length;
    
    //**************************************************************************
    // FIX INDEXES
    
    int dummyIndexes = 0;
    
    if (indexes == NULL) {
        
        indexesLen = databaseLen;
        indexes = (int*) malloc(indexesLen * sizeof(int));
        
        for (int i = 0; i < indexesLen; ++i) {
            indexes[i] = i;
        }
        
        dummyIndexes = 1;
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // SOLVE MULTICARDED
    
    *scores = (int*) malloc(queriesLen * databaseLen * sizeof(int));
    
    int threadNmr = cardsLen;
    int indexesStep = indexesLen / threadNmr;
    
    Thread* threads = (Thread*) malloc((threadNmr - 1) * sizeof(Thread));
    
    KernelContext* contexts = 
        (KernelContext*) malloc(threadNmr * sizeof(KernelContext));
    
    for (int i = 0; i < threadNmr; ++i) {
    
        contexts[i].scores = *scores;
        contexts[i].type = type;
        contexts[i].queries = queries;
        contexts[i].queriesLen = queriesLen;
        contexts[i].longDatabase = longDatabase;
        contexts[i].scorer = scorer;
        contexts[i].card = cards[i];
        contexts[i].indexes = indexes;
        contexts[i].indexesLen = indexesLen;
        
        if (threadNmr < queriesLen) {
            // one query, single card
            contexts[i].queriesStart = i;
            contexts[i].queriesStep = cardsLen;
            contexts[i].indexes = indexes;
            contexts[i].indexesLen = indexesLen;
        } else {
            // one query, multiple cards
            contexts[i].queriesStart = 0;
            contexts[i].queriesStep = 1;
            
            int offset = i * indexesStep;
            contexts[i].indexes = indexes + offset;
            contexts[i].indexesLen = min(indexesStep, indexesLen - offset);
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
    // CLEAN MEMORY

    if (dummyIndexes) {
        free(indexes);    
    }

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
    LongDatabase* longDatabase = context->longDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int card = context->card;
    int queriesStart = context->queriesStart;
    int queriesStep = context->queriesStep;
    
    // set card
    int currentCard;
    CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
    if (currentCard != card) {
        CUDA_SAFE_CALL(cudaThreadExit());
        CUDA_SAFE_CALL(cudaSetDevice(card));
    }

    // prepare gpu db
    LongDatabaseGpu* longDatabaseGpu = longDatabaseGpuCreate(longDatabase, 
        indexes, indexesLen);

    // solve
    for (int i = queriesStart; i < queriesLen; i += queriesStep) {
    
        Chain* query = queries[i];
        int offset = i * longDatabase->length;
        
        kernelSingle(scores + offset, type, query, longDatabase, 
            longDatabaseGpu, scorer);
    }
    
    longDatabaseGpuDelete(longDatabaseGpu);
    
    return NULL;
}

static void kernelSingle(int* scores, int type, Chain* query,
    LongDatabase* longDatabase, LongDatabaseGpu* longDatabaseGpu, 
    Scorer* scorer) {

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    
    int length = longDatabaseGpu->length;

    //**************************************************************************
    // CREATE QUERY PROFILE
    
    int rows = chainGetLength(query);
    int rowsGpu = rows + (4 - rows % 4) % 4;
    
    size_t rowSize = rows * sizeof(char);
    char* row = (char*) malloc(rowSize);
    chainCopyCodes(query, row);

    int subLen = SCORER_MAX_CODE + 1;
    size_t subSize = rowsGpu * subLen * sizeof(char);
    char4* subCpu = (char4*) malloc(subSize);
    memset(subCpu, 0, subSize);
    for (int i = 0; i < rowsGpu / 4; ++i) {
        for (int j = 0; j < SCORER_MAX_CODE; ++j) {
            char4 scr;
            scr.x = i * 4 + 0 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 0], j);
            scr.y = i * 4 + 1 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 1], j);
            scr.z = i * 4 + 2 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 2], j);
            scr.w = i * 4 + 3 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 3], j);
            subCpu[i * subLen + j] = scr;
        }
    }
    
    cudaArray* subArray; 
    CUDA_SAFE_CALL(cudaMallocArray(&subArray, &subTexture.channelDesc, subLen, rowsGpu)); 
    CUDA_SAFE_CALL(cudaMemcpyToArray (subArray, 0, 0, subCpu, subSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTextureToArray(subTexture, subArray));
    subTexture.addressMode[0] = cudaAddressModeClamp;
    subTexture.addressMode[1] = cudaAddressModeClamp;
    subTexture.filterMode = cudaFilterModePoint;
    subTexture.normalized = false;

    //**************************************************************************
    
    //**************************************************************************
    // INIT GPU
    
    int iters = rowsGpu / (THREADS * 4) + (rowsGpu % (THREADS * 4) != 0);
    
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rows_, &rows, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rowsPadded_, &rowsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapOpen_, &gapOpen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapExtend_, &gapExtend, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(length_, &length, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(iters_, &iters, sizeof(int)));
    
    //**************************************************************************

    //**************************************************************************
    // SOVLE
    
    char* codes = longDatabaseGpu->codes;
    int2* hBus = longDatabaseGpu->hBus;
    int3* data = longDatabaseGpu->data;
    int* scoresGpu = longDatabaseGpu->scores;

    void (*function)(int*, char*, int2*, int3*);
    switch (type) {
    case SW_ALIGN: 
        function = swSolve;
        break;
    case NW_ALIGN: 
        function = nwSolve;
        break;
    case HW_ALIGN:
        function = hwSolve;
        break;
    default:
        ERROR("Wrong align type");
    }
    
    function<<<BLOCKS, THREADS>>>(scoresGpu, codes, hBus, data);

    size_t scoresSize = longDatabaseGpu->scoresSize;
    CUDA_SAFE_CALL(cudaMemcpy(scores, scoresGpu, scoresSize, FROM_GPU));
    
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

static LongDatabaseGpu* longDatabaseGpuCreate(LongDatabase* longDatabase,
    int* indexes, int indexesLen) {
    
    size_t dataSize = indexesLen * sizeof(int3);
    int3* data = (int3*) malloc(dataSize);
    
    for (int i = 0; i < indexesLen; ++i) {
        data[i].x = indexes[i];
        data[i].y = longDatabase->starts[indexes[i]];
        data[i].z = longDatabase->lengths[indexes[i]];
    }

    int3* dataGpu;
    CUDA_SAFE_CALL(cudaMalloc(&dataGpu, dataSize));
    CUDA_SAFE_CALL(cudaMemcpy(dataGpu, data, dataSize, TO_GPU));
    
    free(data);
    
    size_t codesSize = longDatabase->codesSize;
    char* codesGpu;
    CUDA_SAFE_CALL(cudaMalloc(&codesGpu, codesSize));
    CUDA_SAFE_CALL(cudaMemcpy(codesGpu, longDatabase->codes, codesSize, TO_GPU));

    size_t hBusSize = longDatabase->codesLen * sizeof(int2);
    int2* hBus;
    CUDA_SAFE_CALL(cudaMalloc(&hBus, hBusSize));

    size_t scoresSize = longDatabase->length * sizeof(int);
    int* scores = (int*) malloc(scoresSize);
    int* scoresGpu;
    CUDA_SAFE_CALL(cudaMalloc(&scoresGpu, scoresSize));
    
    // init scores 
    for (int i = 0; i < longDatabase->length; ++i) {
        scores[i] = NO_SCORE;
    }
    CUDA_SAFE_CALL(cudaMemcpy(scoresGpu, scores, scoresSize, TO_GPU));
    
    free(scores);
    
    LongDatabaseGpu* longDatabaseGpu = 
        (LongDatabaseGpu*) malloc(sizeof(struct LongDatabaseGpu));
    
    longDatabaseGpu->length = indexesLen;
    longDatabaseGpu->codes = codesGpu;
    longDatabaseGpu->data = dataGpu;
    longDatabaseGpu->hBus = hBus;
    longDatabaseGpu->scores = scoresGpu;
    longDatabaseGpu->scoresSize = scoresSize;
    
    return longDatabaseGpu;
}

static void longDatabaseGpuDelete(LongDatabaseGpu* longDatabaseGpu) {

    CUDA_SAFE_CALL(cudaFree(longDatabaseGpu->codes));
    CUDA_SAFE_CALL(cudaFree(longDatabaseGpu->data));
    CUDA_SAFE_CALL(cudaFree(longDatabaseGpu->hBus));
    CUDA_SAFE_CALL(cudaFree(longDatabaseGpu->scores));

    free(longDatabaseGpu);
    longDatabaseGpu = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

__global__ void hwSolve(int* scores, char* codes, int2* hBus, int3* data) {

    for (int i = blockIdx.x; i < length_; i += gridDim.x) {
        hwSolveSingle(scores, codes, hBus, data[i]);
    }
}

__global__ void nwSolve(int* scores, char* codes, int2* hBus, int3* data) {

    for (int i = blockIdx.x; i < length_; i += gridDim.x) {
        nwSolveSingle(scores, codes, hBus, data[i]);
    }
}

__global__ void swSolve(int* scores, char* codes, int2* hBus, int3* data) {

    for (int i = blockIdx.x; i < length_; i += gridDim.x) {
        swSolveSingle(scores, codes, hBus, data[i]);
    }
}

__device__ static int gap(int index) {
    return (-gapOpen_ - index * gapExtend_) * (index >= 0);
}

__device__ void hwSolveSingle(int* scores, char* codes, int2* hBus, int3 data) {

    __shared__ int scoresShr[MAX_THREADS];

    __shared__ int hBusScrShr[MAX_THREADS + 1];
    __shared__ int hBusAffShr[MAX_THREADS + 1];

    int id = data.x;
    int off = data.y;
    int cols = data.z;

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (blockDim.x - 1);
    int col = -threadIdx.x;
    int row = threadIdx.x * 4;
    int iter = 0;
    
    Atom atom;
    atom.mch = gap(row - 1);
    atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
    atom.lAff = INT4_SCORE_MIN;
    
    hBusScrShr[threadIdx.x] = 0;
    hBusAffShr[threadIdx.x] = SCORE_MIN;
    
    for (int i = 0; i < width; ++i) {
    
        int del;
        int valid = col >= 0 && row < rowsPadded_;
    
        if (valid) {
        
            if (iter != 0 && threadIdx.x == 0) {
                atom.up = hBus[off + col];
            } else {
                atom.up.x = hBusScrShr[threadIdx.x];
                atom.up.y = hBusAffShr[threadIdx.x];
            }
            
            char code = codes[off + col];
            char4 rowScores = tex2D(subTexture, code, row >> 2);
            
            del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
            int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
            int mch = atom.mch + rowScores.x;

            atom.rScr.x = MAX3(mch, del, ins);
            atom.rAff.x = ins;

            del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
            mch = atom.lScr.x + rowScores.y;

            atom.rScr.y = MAX3(mch, del, ins);
            atom.rAff.y = ins;
            
            del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
            mch = atom.lScr.y + rowScores.z;

            atom.rScr.z = MAX3(mch, del, ins);
            atom.rAff.z = ins;

            del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
            mch = atom.lScr.z + rowScores.w;

            atom.rScr.w = MAX3(mch, del, ins);
            atom.rAff.w = ins;

            if (row + 0 == rows_ - 1) score = max(score, atom.rScr.x);
            if (row + 1 == rows_ - 1) score = max(score, atom.rScr.y);
            if (row + 2 == rows_ - 1) score = max(score, atom.rScr.z);
            if (row + 3 == rows_ - 1) score = max(score, atom.rScr.w);

            atom.mch = atom.up.x;   
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }
        
        __syncthreads();

        if (valid) {
            if (iter < iters_ - 1 && threadIdx.x == blockDim.x - 1) {
                VEC2_ASSIGN(hBus[off + col], make_int2(atom.rScr.w, del));
            } else {
                hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
                hBusAffShr[threadIdx.x + 1] = del;
            }
        }
        
        col++;
        
        if (col == cols) {

            col = 0;
            row += blockDim.x * 4;
            iter++;
            
            atom.mch = gap(row - 1);
            atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));;
            atom.lAff = INT4_SCORE_MIN;
        }
        
        __syncthreads();
    }

    // write all scores    
    scoresShr[threadIdx.x] = score;
    __syncthreads();
    
    // gather scores
    if (threadIdx.x == 0) {
    
        for (int i = 1; i < blockDim.x; ++i) {
            score = max(score, scoresShr[i]);
        }
    
        scores[id] = score;
    }
}

__device__ void nwSolveSingle(int* scores, char* codes, int2* hBus, int3 data) {

    __shared__ int scoresShr[MAX_THREADS];

    __shared__ int hBusScrShr[MAX_THREADS + 1];
    __shared__ int hBusAffShr[MAX_THREADS + 1];

    int id = data.x;
    int off = data.y;
    int cols = data.z;

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (blockDim.x - 1);
    int col = -threadIdx.x;
    int row = threadIdx.x * 4;
    int iter = 0;
    
    Atom atom;
    atom.mch = gap(row - 1);
    atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
    atom.lAff = INT4_SCORE_MIN;
    
    hBusScrShr[threadIdx.x] = gap(off);
    hBusAffShr[threadIdx.x] = SCORE_MIN;
    
    for (int i = 0; i < width; ++i) {
    
        int del;
        int valid = col >= 0 && row < rowsPadded_;
    
        if (valid) {
        
            if (iter != 0 && threadIdx.x == 0) {
                if (iter == 0) {
                   atom.up.x = gap(off);
                   atom.up.y = SCORE_MIN;
                } else {
                    atom.up = hBus[off + col];
                }
            } else {
                atom.up.x = hBusScrShr[threadIdx.x];
                atom.up.y = hBusAffShr[threadIdx.x];
            }
            
            char code = codes[off + col];
            char4 rowScores = tex2D(subTexture, code, row >> 2);
            
            del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
            int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
            int mch = atom.mch + rowScores.x;

            atom.rScr.x = MAX3(mch, del, ins);
            atom.rAff.x = ins;

            del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
            mch = atom.lScr.x + rowScores.y;

            atom.rScr.y = MAX3(mch, del, ins);
            atom.rAff.y = ins;
            
            del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
            mch = atom.lScr.y + rowScores.z;

            atom.rScr.z = MAX3(mch, del, ins);
            atom.rAff.z = ins;

            del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
            mch = atom.lScr.z + rowScores.w;

            atom.rScr.w = MAX3(mch, del, ins);
            atom.rAff.w = ins;

            atom.mch = atom.up.x;   
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }
        
        __syncthreads();

        if (valid) {
            if (iter < iters_ - 1 && threadIdx.x == blockDim.x - 1) {
                VEC2_ASSIGN(hBus[off + col], make_int2(atom.rScr.w, del));
            } else {
                hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
                hBusAffShr[threadIdx.x + 1] = del;
            }
        }
        
        col++;
        
        if (col == cols) {

            if (row + 0 == rows_ - 1) score = max(score, atom.lScr.x);
            if (row + 1 == rows_ - 1) score = max(score, atom.lScr.y);
            if (row + 2 == rows_ - 1) score = max(score, atom.lScr.z);
            if (row + 3 == rows_ - 1) score = max(score, atom.lScr.w);
            
            col = 0;
            row += blockDim.x * 4;
            iter++;
            
            atom.mch = gap(row - 1);
            atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));;
            atom.lAff = INT4_SCORE_MIN;
        }
        
        __syncthreads();
    }

    // write all scores    
    scoresShr[threadIdx.x] = score;
    __syncthreads();
    
    // gather scores
    if (threadIdx.x == 0) {
    
        for (int i = 1; i < blockDim.x; ++i) {
            score = max(score, scoresShr[i]);
        }
    
        scores[id] = score;
    }
}

__device__ void swSolveSingle(int* scores, char* codes, int2* hBus, int3 data) {

    __shared__ int scoresShr[MAX_THREADS];

    __shared__ int hBusScrShr[MAX_THREADS + 1];
    __shared__ int hBusAffShr[MAX_THREADS + 1];

    int id = data.x;
    int off = data.y;
    int cols = data.z;

    int score = 0;
    
    int width = cols * iters_ + 2 * (blockDim.x - 1);
    int col = -threadIdx.x;
    int row = threadIdx.x * 4;
    int iter = 0;
    
    Atom atom;
    atom.mch = 0;
    atom.lScr = INT4_ZERO;
    atom.lAff = INT4_SCORE_MIN;
    
    hBusScrShr[threadIdx.x] = 0;
    hBusAffShr[threadIdx.x] = SCORE_MIN;
    
    for (int i = 0; i < width; ++i) {
    
        int del;
        int valid = col >= 0 && row < rowsPadded_;
    
        if (valid) {
        
            if (iter != 0 && threadIdx.x == 0) {
                atom.up = hBus[off + col];
            } else {
                atom.up.x = hBusScrShr[threadIdx.x];
                atom.up.y = hBusAffShr[threadIdx.x];
            }
            
            char code = codes[off + col];
            char4 rowScores = tex2D(subTexture, code, row >> 2);
            
            del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
            int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
            int mch = atom.mch + rowScores.x;

            atom.rScr.x = MAX4(0, mch, del, ins);
            atom.rAff.x = ins;

            del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
            mch = atom.lScr.x + rowScores.y;

            atom.rScr.y = MAX4(0, mch, del, ins);
            atom.rAff.y = ins;
            
            del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
            mch = atom.lScr.y + rowScores.z;

            atom.rScr.z = MAX4(0, mch, del, ins);
            atom.rAff.z = ins;

            del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
            mch = atom.lScr.z + rowScores.w;

            atom.rScr.w = MAX4(0, mch, del, ins);
            atom.rAff.w = ins;

            score = max(score, atom.rScr.x);
            score = max(score, atom.rScr.y);
            score = max(score, atom.rScr.z);
            score = max(score, atom.rScr.w);

            atom.mch = atom.up.x;   
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }
        
        __syncthreads();

        if (valid) {
            if (iter < iters_ - 1 && threadIdx.x == blockDim.x - 1) {
                VEC2_ASSIGN(hBus[off + col], make_int2(atom.rScr.w, del));
            } else {
                hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
                hBusAffShr[threadIdx.x + 1] = del;
            }
        }
        
        col++;
        
        if (col == cols) {

            col = 0;
            row += blockDim.x * 4;
            iter++;
                    
            atom.mch = 0;
            atom.lScr = INT4_ZERO;
            atom.lAff = INT4_SCORE_MIN;
        }
        
        __syncthreads();
    }

    // write all scores    
    scoresShr[threadIdx.x] = score;
    __syncthreads();
    
    // gather scores
    if (threadIdx.x == 0) {
    
        for (int i = 1; i < blockDim.x; ++i) {
            score = max(score, scoresShr[i]);
        }
    
        scores[id] = score;
    }
}

//------------------------------------------------------------------------------

//******************************************************************************
