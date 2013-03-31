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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "error.h"
#include "scorer.h"
#include "utils.h"

#include "pre_proc.h"

#define SCORERS_LEN (sizeof(scorers) / sizeof(ScorerEntry))

typedef struct ScorerEntry {
    const char* name;
    int (*table)[26][26];
} ScorerEntry;

// to register a scorer just add his name and corresponding table to this array
static ScorerEntry scorers[] = {
    { BLOSUM_62, &BLOSUM_62_TABLE }, // default one
    { BLOSUM_45, &BLOSUM_45_TABLE },
    { BLOSUM_50, &BLOSUM_50_TABLE },
    { BLOSUM_80, &BLOSUM_80_TABLE },
    { BLOSUM_90, &BLOSUM_90_TABLE },
    { PAM_30, &PAM_30_TABLE },
    { PAM_70, &PAM_70_TABLE },
    { PAM_250, &PAM_250_TABLE }
};

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CHAIN UTILS

extern Chain* createChainComplement(Chain* chain) {

    int length = chainGetLength(chain);
    char* string = (char*) malloc(length * sizeof(char));
    
    int i;
    for (i = 0; i < length; ++i) {
    
        char chr = chainGetChar(chain, i);
        
        switch(chr) {
            case 'A':
                chr = 'T';
                break;
            case 'T':
                chr = 'A';
                break;
            case 'C':
                chr = 'G';
                break;     
            case 'G':
                chr = 'C';
                break;       
        }
        
        string[length - 1 - i] = chr;
    }
    
    char* name = "CMPL";
    
    return chainCreate(name, 4, string, length);
}

extern void readFastaChain(Chain** chain, const char* path) {

    FILE* f = fileSafeOpen(path, "r");
    
    char* str = (char*) malloc(fileLength(f) * sizeof(char));
    int strLen = 0;
    
    char* name = (char*) malloc(1024 * sizeof(char));
    int nameLen = 0;
    
    char buffer[4096];
    int isName = 1;
    
    while (!feof(f)) {
        
        int read = fread(buffer, sizeof(char), 4096, f);
        
        int i;
        for (i = 0; i < read; ++i) {
            
            char c = buffer[i];
            
            if (isName) {
                if (c == '\n') {
                    name[nameLen] = 0;
                    isName = 0;
                } else if (!(nameLen == 0 && c == '>')) {
                    name[nameLen++] = c;                
                }
            } else if (isalpha(c)) {
                str[strLen++] = c;
            }
        }
    }
    
    *chain = chainCreate(name, nameLen, str, strLen);

    free(str);
    free(name);
    
    fclose(f);
}

extern void readFastaChains(Chain*** chains_, int* chainsLen_, const char* path) {

    TIMER_START("Reading database");
    
    FILE* f = fileSafeOpen(path, "r");
    
    int strSize = 4096;
    char* str = (char*) malloc(strSize * sizeof(char));
    int strLen = 0;
    
    char* name = (char*) malloc(1024 * sizeof(char));
    int nameLen = 0;
    
    char buffer[4096];
    int isName = 1;
    
    int chainsSize = 1000;
    int chainsLen = 0;
    Chain** chains = (Chain**) malloc(chainsSize * sizeof(Chain*));
    
    while (!feof(f)) {
        
        int read = fread(buffer, sizeof(char), 4096, f);
        
        int i;
        for (i = 0; i < read; ++i) {
            
            char c = buffer[i];
            
            if (!isName && c == '>') {
            
                isName = 1;
                
                Chain* chain = chainCreate(name, nameLen, str, strLen);
                
                if (chainsLen + 1 == chainsSize) {
                    chainsSize *= 2;
                    chains = (Chain**) realloc(chains, chainsSize * sizeof(Chain*));
                }
                chains[chainsLen++] = chain;
                      
                nameLen = 0;
                strLen = 0;
            }
            
            if (isName) {
                if (c == '\n') {
                    name[nameLen] = 0;
                    isName = 0;
                } else if (!(nameLen == 0 && c == '>')) {
                    name[nameLen++] = c;                
                }
            } else if (isalpha(c)) {
                if (strLen == strSize) {
                    strSize *= 2;
                    str = (char*) realloc(str, strSize * sizeof(char));
                }
                str[strLen++] = c;
            }
        }
    }
    
    Chain* chain = chainCreate(name, nameLen, str, strLen);
    chains[chainsLen++] = chain;
    
    *chainsLen_ = chainsLen;
    *chains_ = chains;
    
    free(str);
    free(name);
    
    fclose(f);
    
    TIMER_STOP;
}

extern void deleteFastaChains(Chain** chains, int chainsLen) {

    int i;
    for (i = 0; i < chainsLen; ++i) {
        chainDelete(chains[i]);
    }
    
    free(chains);
    chains = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SCORES UTILS

extern void scorerCreateConst(Scorer** scorer, int match, int mismatch, 
    int gapOpen, int gapExtend) {
    
    int scores[SCORER_MAX_CODE][SCORER_MAX_CODE];
    
    int i, j;
    for (i = 0; i < SCORER_MAX_CODE; ++i) {
        for (j = 0; j < SCORER_MAX_CODE; ++j) {
            scores[i][j] = i == j ? match : mismatch;
        }
    }
    
    *scorer = scorerCreate("CONST", scores, gapOpen, gapExtend);
}

extern void scorerCreateMatrix(Scorer** scorer, char* name, int gapOpen, 
    int gapExtend) {
    
    int index = -1;
  
    int i;
    for (i = 0; i < SCORERS_LEN; ++i) {
        if (strcmp(name, scorers[i].name) == 0) {
            index = i;
            break;
        }
    }
    
    if (index == -1) {
        index = 0;
        WARNING(1, "unknown table %s, using %s", name, scorers[index].name);
    }
    
    ScorerEntry* entry = &(scorers[index]);
    *scorer = scorerCreate(entry->name, *(entry->table), gapOpen, gapExtend);
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************
