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

#ifndef __SWSHARP_SCORERH__
#define __SWSHARP_SCORERH__

#ifdef __cplusplus 
extern "C" {
#endif

#define SCORER_MAX_CODE 26

typedef struct Scorer Scorer;

extern Scorer* scorerCreate(const char* name, 
    int scores[SCORER_MAX_CODE][SCORER_MAX_CODE], int gapOpen, int gapExtend);
    
extern void scorerDelete(Scorer* scorer);

extern int scorerGetGapExtend(Scorer* scorer);
extern int scorerGetGapOpen(Scorer* scorer);
extern int scorerGetMaxScore(Scorer* scorer);
extern char* scorerGetName(Scorer* scorer);
extern int scorerIsScalar(Scorer* scorer);

extern int scorerScore(Scorer* scorer, char a, char b);

extern Scorer* scorerDeserialize(char* bytes);

extern void scorerSerialize(char** bytes, int* bytesLen, Scorer* scorer);

#ifdef __cplusplus 
}
#endif
#endif // __SWSHARP_SCORERH__
