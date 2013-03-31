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

#ifndef __SW_SHARP_PRE_PROCESH__
#define __SW_SHARP_PRE_PROCESH__

#include "chain.h"
#include "scorer.h"

extern Chain* createChainComplement(Chain* chain);

extern void readFastaChain(Chain** chain, const char* path);
extern void readFastaChains(Chain*** chains, int* chainsLen, const char* path);
extern void deleteFastaChains(Chain** chains, int chainsLen);

extern void scorerCreateConst(Scorer** scorer, int match, int mismatch, 
    int gapOpen, int gapExtend);
    
extern void scorerCreateMatrix(Scorer** scorer, char* name, int gapOpen, 
    int gapExtend);

#endif // __SW_SHARP_PRE_PROCESH__
