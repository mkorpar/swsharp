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

#ifndef __SW_SHARP_POST_PROCESH__
#define __SW_SHARP_POST_PROCESH__

#include "alignment.h"
#include "db_alignment.h"

#ifdef __cplusplus 
extern "C" {
#endif


#define SW_OUT_PAIR         0
#define SW_OUT_PLOT         1
#define SW_OUT_STAT         2
#define SW_OUT_STAT_PAIR    3
#define SW_OUT_DUMP         4

#define SW_OUT_DB_BLASTM1   0
#define SW_OUT_DB_BLASTM8   1
#define SW_OUT_DB_BLASTM9   2
#define SW_OUT_DB_LIGHT     3

extern int checkAlignment(Alignment* alignment);

extern Alignment* readAlignment(char* path);

extern void outputAlignment(Alignment* alignment, char* path, int type);

extern void outputDatabase(DbAlignment** dbAlignments, int dbAlignmentsLen, 
    char* path, int type);
    
extern void outputShotgunDatabase(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen, char* path, int type);
    
extern void deleteShotgunDatabase(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_POST_PROCESH__
