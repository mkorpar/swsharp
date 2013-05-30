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
    int (*table)[26 * 26];
} ScorerEntry;

// to register a scorer just add his name and corresponding table to this array
static ScorerEntry scorers[] = {
    { "BLOSUM_62", &BLOSUM_62_TABLE }, // default one
    { "BLOSUM_45", &BLOSUM_45_TABLE },
    { "BLOSUM_50", &BLOSUM_50_TABLE },
    { "BLOSUM_80", &BLOSUM_80_TABLE },
    { "BLOSUM_90", &BLOSUM_90_TABLE },
    { "PAM_30", &PAM_30_TABLE },
    { "PAM_70", &PAM_70_TABLE },
    { "PAM_250", &PAM_250_TABLE }
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
    
    char name[1000];
    sprintf(name, "complement: %s", chainGetName(chain));
    
    int nameLen = strlen(name);

    Chain* complement = chainCreate(name, nameLen, string, length);
    
    free(string);
    
    return complement;
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
                } else if (!(nameLen == 0 && (c == '>' || isspace(c)))) {
                    if (c != '\r') {
                        name[nameLen++] = c;
                    }         
                }
            } else {
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
                } else if (!(nameLen == 0 && (c == '>' || isspace(c)))) {
                    if (c != '\r') {
                        name[nameLen++] = c;
                    }              
                }
            } else {
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

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SCORES UTILS

extern void scorerCreateScalar(Scorer** scorer, int match, int mismatch, 
    int gapOpen, int gapExtend) {
    
    int scores[26 * 26];
    
    int i, j;
    for (i = 0; i < 26; ++i) {
        for (j = 0; j < 26; ++j) {
            scores[i * 26 + j] = i == j ? match : mismatch;
        }
    }
    
    char name[100];
    sprintf(name, "match/mismatch +%d/%d", match, mismatch);
    
    *scorer = scorerCreate(name, scores, 26, gapOpen, gapExtend);
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
    
    ASSERT(index != -1, "unknown table %s", name);
    
    ScorerEntry* entry = &(scorers[index]);
    *scorer = scorerCreate(entry->name, *(entry->table), 26, gapOpen, gapExtend);
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************
