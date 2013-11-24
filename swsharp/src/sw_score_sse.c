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
#ifdef __SSE2__

#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "utils.h"

#include "sse_module.h"

typedef struct HBus {
    int scr;
    int aff;
} HBus;

//******************************************************************************
// PUBLIC

extern int swScoreSse(Chain* query, Chain* target, Scorer* scorer);

//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************

//******************************************************************************
// PUBLIC

extern int swScoreSse(Chain* query, Chain* target, Scorer* scorer) {

    if (scorerGetMaxScore(scorer) <= 0) {
        return 0;
    }

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int max = 0;
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
    memset(hBus, 0, cols * sizeof(HBus));

    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (row = 0; row < rows; ++row) {
    
        int iScr = 0;
        int iAff = SCORE_MIN;
        
        int diag = 0;
        
        for (col = 0; col < cols; ++col) {
        
            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
            // DELETE END
            
            int scr = MAX(MAX(0, mch), MAX(ins, del));
           
            max = MAX(max, scr);
            
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }

    free(hBus);

    return max;
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************
#endif // __SSE2__
