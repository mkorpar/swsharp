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

#include "error.h"

#include "cuda_utils.h"

extern void cudaGetCards(int** cards, int* cardsLen) {
    
    cudaGetDeviceCount(cardsLen);
    
    *cards = (int*) malloc(*cardsLen * sizeof(int));
    
    for (int i = 0; i < *cardsLen; ++i) {
        (*cards)[i] = i;   
    }
}

extern int cudaCheckCards(int* cards, int cardsLen) {
    
    int maxDeviceId;
    cudaGetDeviceCount(&maxDeviceId);
    
    for (int i = 0; i < cardsLen; ++i) {
        if (cards[i] >= maxDeviceId) {
            return 0;
        }   
    }
    
    return 1;
}

extern void cudaCardBuckets(int*** cardBuckets, int** cardBucketsLens, 
    int* cards, int cardsLen, int buckets) {
    
    ASSERT(buckets <= cardsLen && buckets >= 1, "invalid bucket data");
    
    *cardBuckets = (int**) malloc(buckets * sizeof(int*));
    *cardBucketsLens = (int*) malloc(buckets * sizeof(int));
    
    memset(*cardBucketsLens, 0, buckets * sizeof(int));
    
    int i;
    
    int cardsLeft = cardsLen;
    i = 0;
    while (cardsLeft > 0) {
        (*cardBucketsLens)[i]++;
        i = (i + 1) % buckets;
        cardsLeft--;
    }
    
    int offset = 0;
    for (i = 0; i < buckets; ++i) {
       (*cardBuckets)[i] = cards + offset;
       offset += (*cardBucketsLens)[i];
    }
}
