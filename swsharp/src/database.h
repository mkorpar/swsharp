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
/**
@file

@brief
*/

#ifndef __SW_SHARP_DATABASEH__
#define __SW_SHARP_DATABASEH__

#include "chain.h"
#include "constants.h"
#include "dbAlignment.h"
#include "scorer.h"
#include "thread.h"

#ifdef __cplusplus 
extern "C" {
#endif

typedef struct ChainDatabase ChainDatabase;

typedef void (*ValueFunction)(float* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, void* param);

extern ChainDatabase* chainDatabaseCreate(Chain** database, int databaseLen);

extern void chainDatabaseDelete(ChainDatabase* chainDatabase);

extern void alignDatabase(DbAlignment*** dbAlignments, int* dbAlignmentsLen, 
    int type, Chain* query, ChainDatabase* chainDatabase, Scorer* scorer, 
    int maxAlignments, ValueFunction valueFunction, void* valueFunctionParam, 
    float valueThreshold, int* indexes, int indexesLen, int* cards, 
    int cardsLen, Thread* thread);
    
extern void shotgunDatabase(DbAlignment**** dbAlignments, int** dbAlignmentsLen, 
    int type, Chain** queries, int queriesLen, ChainDatabase* chainDatabase, 
    Scorer* scorer, int maxAlignments, ValueFunction valueFunction, 
    void* valueFunctionParam, float valueThreshold, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_DATABASEH__
