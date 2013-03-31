/*
swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and 
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Å ikiÄ‡

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

#include "utils.h"

#include "scorer.h"

struct Scorer {

    char* name;
    int nameLen;
    
    int gapOpen;
    int gapExtend;
    int table[SCORER_MAX_CODE][SCORER_MAX_CODE];
    
    int maxScore;
    int scalar;
};

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

static int isScalar(Scorer* scorer);

static int maxScore(Scorer* scorer);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern Scorer* scorerCreate(const char* name, 
    int scores[SCORER_MAX_CODE][SCORER_MAX_CODE], int gapOpen, int gapExtend) {

    Scorer* scorer = (Scorer*) malloc(sizeof(struct Scorer));

    int nameLen = strlen(name) + 1;
    scorer->name = (char*) malloc(nameLen * sizeof(char));
    memcpy(scorer->name, name, (nameLen - 1) * sizeof(char));
    scorer->name[nameLen - 1] = '\0';
    
    scorer->nameLen = nameLen;
    
    memcpy(scorer->table, scores, SCORER_MAX_CODE * SCORER_MAX_CODE * sizeof(int));

    scorer->gapOpen = gapOpen;
    scorer->gapExtend = gapExtend;

    scorer->maxScore = maxScore(scorer);
    scorer->scalar = isScalar(scorer);
    
    return scorer;
}

extern void scorerDelete(Scorer* scorer) {
    free(scorer->name);
    free(scorer);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GETTERS

extern int scorerGetGapExtend(Scorer* scorer) {
    return scorer->gapExtend;
}

extern int scorerGetGapOpen(Scorer* scorer) {
    return scorer->gapOpen;
}

extern int scorerGetMaxScore(Scorer* scorer) {
    return scorer->maxScore;
}

extern char* scorerGetName(Scorer* scorer) {
    return scorer->name;
}

extern int scorerIsScalar(Scorer* scorer) {
    return scorer->scalar;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// FUNCTIONS

extern int scorerScore(Scorer* scorer, char a, char b) {
    return scorer->table[(int) a][(int) b];
}

extern Scorer* scorerDeserialize(char* bytes) {

    int ptr = 0;
    
    int nameLen;
    memcpy(&nameLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    char* name = (char*) malloc(nameLen);
    memcpy(name, bytes + ptr, nameLen);
    ptr += nameLen;
    
    int gapOpen;
    memcpy(&gapOpen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    int gapExtend;
    memcpy(&gapExtend, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    int table[SCORER_MAX_CODE][SCORER_MAX_CODE];
    memcpy(table, bytes + ptr, sizeof(table));
    ptr += sizeof(table);
    
    int maxScore;
    memcpy(&maxScore, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    int scalar;
    memcpy(&scalar, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    Scorer* scorer = (Scorer*) malloc(sizeof(struct Scorer));
    
    scorer->nameLen = nameLen;
    scorer->name = name;
    scorer->gapOpen = gapOpen;
    scorer->gapExtend = gapExtend;
    scorer->maxScore = maxScore;
    scorer->scalar = scalar;
    
    memcpy(scorer->table, table, sizeof(table));
    
    return scorer;
}

extern void scorerSerialize(char** bytes, int* bytesLen, Scorer* scorer) {

    *bytesLen = 0;
    *bytesLen += sizeof(int); // nameLen
    *bytesLen += scorer->nameLen; // name
    *bytesLen += sizeof(int); // gapOpen
    *bytesLen += sizeof(int); // gapExtend
    *bytesLen += sizeof(scorer->table); // table
    *bytesLen += sizeof(int); // maxScore
    *bytesLen += sizeof(int); // scalar

    *bytes = (char*) malloc(*bytesLen);
        
    int ptr = 0;
    
    memcpy(*bytes + ptr, &scorer->nameLen, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, scorer->name, scorer->nameLen);
    ptr += scorer->nameLen;
    
    memcpy(*bytes + ptr, &scorer->gapOpen, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, &scorer->gapExtend, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, scorer->table, sizeof(scorer->table));
    ptr += sizeof(scorer->table);
    
    memcpy(*bytes + ptr, &scorer->maxScore, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, &scorer->scalar, sizeof(int));
    ptr += sizeof(int);
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

static int isScalar(Scorer* scorer) {
    
    int x, i, j;
    
    x = scorer->table[0][0];
    for (i = 1; i < SCORER_MAX_CODE; ++i) {
        if (scorer->table[i][i] != x) {
            return 0;
        }
    }
    
    x = scorer->table[0][1];
    for (i = 0; i < SCORER_MAX_CODE; ++i) {
        for (j = 0; j < SCORER_MAX_CODE; ++j) {
            if (i != j && scorer->table[i][j] != x) {
                return 0;
            }
        }
    }
    
    return 1;
}

static int maxScore(Scorer* scorer) {
    
    int i, j;
    
    int max = scorer->table[0][0];
    for (i = 0; i < SCORER_MAX_CODE; ++i) {
        for (j = 0; j < SCORER_MAX_CODE; ++j) {
            max = MAX(max, scorer->table[i][j]);
        }
    }
    
    return max;
}

//******************************************************************************
