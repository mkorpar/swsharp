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
#include "dbAlignment.h"

#ifdef __cplusplus 
extern "C" {
#endif

extern int checkAlignment(Alignment* alignment);

extern void dumpAlignment(Alignment* alignment, char* path);
extern Alignment* readAlignment(const char* path);

extern void outputPair(Alignment* alignment, char* path);

extern void outputStat(Alignment* alignment, char* path);

extern void outputPlot(Alignment* alignment, char* path);

extern void outputDatabase(DbAlignment** dbAlignments, int dbAlignmentsLen, 
    char* path);
    
extern void outputBlastM1(DbAlignment** dbAlignments, int dbAlignmentsLen, 
    char* path);
    
extern void outputBlastM8(DbAlignment** dbAlignments, int dbAlignmentsLen, 
    char* path);
    
extern void outputBlastM9(DbAlignment** dbAlignments, int dbAlignmentsLen, 
    char* path);

extern void outputShotgunDatabase(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen, char* path);
    
extern void outputShotgunBlastM1(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen, char* path);
    
extern void outputShotgunBlastM8(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen, char* path);
    
extern void outputShotgunBlastM9(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen, char* path);

extern void deleteDbAlignements(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_POST_PROCESH__
