#include <stdio.h>

#include "chain.h"
#include "error.h"
#include "scorer.h"

#include "sse_module.h"

extern int scoreDatabaseSse(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer) {

    WARNING(1, "SSE not available");

    return -1;
}
