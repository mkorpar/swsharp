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

#include <stdio.h>

#include "chain.h"
#include "constants.h"
#include "error.h"
#include "scorer.h"

#include "swimd/Swimd.h"

#include "sse_module.h"

extern int scoreDatabaseSse(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer) {

#ifdef __SSE4_1__

    int mode;
    switch (type) {
    case SW_ALIGN:
        mode = SWIMD_MODE_SW;
        break;
    case HW_ALIGN:
        mode = SWIMD_MODE_HW;
        break;
    case NW_ALIGN:
        mode = SWIMD_MODE_NW;
        break;
    case OV_ALIGN:
        mode = SWIMD_MODE_OV;
        break;
    default:
        return -1;
    }

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int* table = (int*) scorerGetTable(scorer);
    int maxCode = scorerGetMaxCode(scorer);

    unsigned char* queryPtr = (unsigned char*) chainGetCodes(query);
    int queryLen = chainGetLength(query);

    unsigned char** databasePtrs = 
        (unsigned char**) malloc(databaseLen * sizeof(unsigned char*));

    int* databaseLens = (int*) malloc(databaseLen * sizeof(int));

    int i;
    for (i = 0; i < databaseLen; ++i) {
        databasePtrs[i] = (unsigned char*) chainGetCodes(database[i]);
        databaseLens[i] = chainGetLength(database[i]);
    }

    int status = swimdSearchDatabase(queryPtr, queryLen, databasePtrs, 
        databaseLen, databaseLens, gapOpen, gapExtend, table, maxCode, scores, mode);

    free(databasePtrs);
    free(databaseLens);

    return status;

#else
    return -1;
#endif
}
