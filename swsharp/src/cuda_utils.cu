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

#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "utils.h"

#include "cuda_utils.h"

extern void cudaGetCards(int** cards, int* cardsLen) {

#ifdef __CUDACC__
    cudaGetDeviceCount(cardsLen);
    
    *cards = (int*) malloc(*cardsLen * sizeof(int));
    
    for (int i = 0; i < *cardsLen; ++i) {
        (*cards)[i] = i;   
    }
#else
    *cards = NULL;
    *cardsLen = 0;
#endif
}

extern int cudaCheckCards(int* cards, int cardsLen) {

#ifdef __CUDACC__
    int maxDeviceId;
    cudaGetDeviceCount(&maxDeviceId);
    
    for (int i = 0; i < cardsLen; ++i) {
        if (cards[i] >= maxDeviceId) {
            return 0;
        }   
    }
    
    return 1;
#else
    return cardsLen == 0;
#endif
}

extern size_t cudaMinimalGlobalMemory(int* cards, int cardsLen) {

#ifdef __CUDACC__

    if (cards == NULL || cardsLen == 0) {
        return 0;
    }

    size_t minMem = (size_t) -1;
    for (int i = 0; i < cardsLen; ++i) {

        cudaDeviceProp cdp;
        cudaGetDeviceProperties(&cdp, i);

        minMem = MIN(minMem, cdp.totalGlobalMem);
    }

    return minMem;
#else
    return 0;
#endif
}

