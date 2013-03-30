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

#ifndef __SW_SHARP_DBALIGNMENTH__
#define __SW_SHARP_DBALIGNMENTH__

#include "alignment.h"
#include "chain.h"
#include "scorer.h"

#ifdef __cplusplus 
extern "C" {
#endif

typedef struct DbAlignment DbAlignment;

extern DbAlignment* dbAlignmentCreate(Chain* query, int queryStart, int queryEnd,
    int queryIdx, Chain* target, int targetStart, int targetEnd, int targetIdx, 
    float value, int score, Scorer* scorer, char* path, int pathLen);

extern void dbAlignmentDelete(DbAlignment* dbAlignment);

extern char dbAlignmentGetMove(DbAlignment* dbAlignment, int index);
extern int dbAlignmentGetPathLen(DbAlignment* dbAlignment);
extern Chain* dbAlignmentGetQuery(DbAlignment* dbAlignment);
extern int dbAlignmentGetQueryEnd(DbAlignment* dbAlignment);
extern int dbAlignmentGetQueryIdx(DbAlignment* dbAlignment);
extern int dbAlignmentGetQueryStart(DbAlignment* dbAlignment);
extern int dbAlignmentGetScore(DbAlignment* dbAlignment);
extern Scorer* dbAlignmentGetScorer(DbAlignment* dbAlignment);
extern Chain* dbAlignmentGetTarget(DbAlignment* dbAlignment);
extern int dbAlignmentGetTargetEnd(DbAlignment* dbAlignment);
extern int dbAlignmentGetTargetIdx(DbAlignment* dbAlignment);
extern int dbAlignmentGetTargetStart(DbAlignment* dbAlignment);
extern float dbAlignmentGetValue(DbAlignment* dbAlignment);

extern void dbAlignmentCopyPath(DbAlignment* dbAlignment, char* dest);
extern Alignment* dbAlignmentToAlignment(DbAlignment* dbAlignment);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_DBALIGNMENTH__
