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

#include "error.h"
#include "scorer.h"
#include "thread.h"

#include "chain.h"

struct Chain {

    char* name;
    int nameLen;
    
    int length;
    
    char* codes;
    
    int reverseOrigin;
    int reverseCalculated;
    char* reverseCodes;
    Mutex reverseWrite;

    int isView;
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
// CONSTRUCTOR, DESTRUCTOR

extern Chain* chainCreate(char* name, int nameLen, char* string, int stringLen) {

    ASSERT(name != NULL && nameLen > 0 && string != NULL && stringLen > 0, 
        "invalid chain data");

    Chain* chain = (Chain*) malloc(sizeof(struct Chain));

    chain->isView = 0;
    
    chain->nameLen = nameLen + 1;
    chain->name = (char*) malloc((nameLen + 1) * sizeof(char));
    chain->name[nameLen] = 0;
    memcpy(chain->name, name, nameLen * sizeof(char));
    
    chain->codes = (char*) malloc(stringLen * sizeof(char));
    chain->length = 0;
    
    int i;
    for (i = 0; i < stringLen; ++i) {
    
        char code = scorerEncode(string[i]);
        
        if (code != -1) {      
            chain->codes[chain->length] = code;
            chain->length++;
        }
    }
    
    chain->reverseCodes = NULL;
    chain->reverseOrigin = 0;
    chain->reverseCalculated = 0;
    mutexCreate(&(chain->reverseWrite));

    return chain;
}

extern void chainDelete(Chain* chain) {

    if (!chain->isView) {
        mutexDelete(&(chain->reverseWrite));
        free(chain->codes);
        free(chain->name);
    }
    
    if (chain->reverseOrigin) {
        free(chain->reverseCodes);
    }
    
    free(chain); 
    chain = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GETTERS

extern char chainGetChar(Chain* chain, int index) {
    return scorerDecode(chain->codes[index]);
}

extern char chainGetCode(Chain* chain, int index) {
    return chain->codes[index];
}

extern const char* chainGetName(Chain* chain) {
    return chain->name;
}

extern int chainGetLength(Chain* chain) {
    return chain->length;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// FUNCTIONS

extern void chainCreateReverse(Chain* chain) {

    if (chain->reverseCalculated) {
        return;
    }
    
    mutexLock(&(chain->reverseWrite));
    
    if (chain->reverseCalculated) {
        mutexUnlock(&(chain->reverseWrite));
        return;
    }
    
    chain->reverseCodes = (char*) malloc(chain->length * sizeof(char));

    int i;
    for (i = 0; i < chain->length; ++i) {
        chain->reverseCodes[chain->length - i - 1] = chain->codes[i];
    }
    
    chain->reverseOrigin = 1;
    chain->reverseCalculated = 1;

    mutexUnlock(&(chain->reverseWrite));
}

extern Chain* chainCreateView(Chain* chain, int start, int end, int reverse) {

    ASSERT(chain->reverseCalculated, "chain reverse not calculated");

    Chain* view = (Chain*) malloc(sizeof(struct Chain));

    view->length = (end - start) + 1;
    view->isView = 1;
    view->name = chain->name;
    view->reverseOrigin = 0;
    view->reverseCalculated = chain->reverseCalculated;
    view->reverseWrite = chain->reverseWrite;
    
    if (reverse) {
        view->codes = chain->reverseCodes + (chain->length - end - 1);
        view->reverseCodes = chain->codes + start;
    } else {
        view->codes = chain->codes + start;
        view->reverseCodes = chain->reverseCodes + (chain->length - end - 1);
    }

    return view;
}

extern void chainCopyCodes(Chain* chain, char* dest) {
    memcpy(dest, chain->codes, chain->length * sizeof(char));
}

extern Chain* chainDeserialize(char* bytes) {
    
    int ptr = 0;
    
    int nameLen;
    memcpy(&nameLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    char* name = (char*) malloc(nameLen);
    memcpy(name, bytes + ptr, nameLen);
    ptr += nameLen;
    
    int length;
    memcpy(&length, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    char* codes = (char*) malloc(length);
    memcpy(codes, bytes + ptr, length);
    ptr += length;
    
    Chain* chain = (Chain*) malloc(sizeof(struct Chain));
    
    chain->name = name;
    chain->nameLen = nameLen;
    chain->length = length;
    chain->codes = codes;
    chain->isView = 0;
    
    chain->reverseCodes = NULL;
    chain->reverseOrigin = 0;
    chain->reverseCalculated = 0;
    mutexCreate(&(chain->reverseWrite));
    
    return chain;
}

extern void chainSerialize(char** bytes, int* bytesLen, Chain* chain) {

    ASSERT(!chain->isView, "chain view cannot be serialized");
    
    *bytesLen = 0;
    *bytesLen += sizeof(int); // nameLen
    *bytesLen += chain->nameLen; // name
    *bytesLen += sizeof(int); // length
    *bytesLen += chain->length; // codes
    
    *bytes = (char*) malloc(*bytesLen);

    int ptr = 0;
    
    memcpy(*bytes + ptr, &chain->nameLen, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, chain->name, chain->nameLen);
    ptr += chain->nameLen;
    
    memcpy(*bytes + ptr, &chain->length, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, chain->codes, chain->length);
    ptr += chain->length;
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************
