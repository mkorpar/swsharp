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

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "swsharp/swsharp.h"

#include "database_utils.h"

using namespace std;

//******************************************************************************
// PUBLIC

extern void joinShotgunDatabases(DbAlignment**** dbAlignments, 
    int** dbAlignmentsLens, int* dbAlignmentsLen, DbAlignment**** dbAlignmentsMpi, 
    int** dbAlignmentsLensMpi, int* dbAlignmentsLenMpi, int databasesLen, 
    int maxAlignments);
    
//******************************************************************************

//******************************************************************************
// PRIVATE

static bool dbAlignmentCmp(DbAlignment* a, DbAlignment* b);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void joinShotgunDatabases(DbAlignment**** dbAlignments_, 
    int** dbAlignmentsLens_, int* dbAlignmentsLen_, 
    DbAlignment**** dbAlignmentsMpi, int** dbAlignmentsLensMpi, 
    int* dbAlignmentsLenMpi, int databasesLen, int maxAlignments) {
    
    map<int, vector<DbAlignment*> > data;
    
    for (int i = 0; i < databasesLen; ++i) {
        for (int j = 0; j < dbAlignmentsLenMpi[i]; ++j) {
            for (int k = 0; k < dbAlignmentsLensMpi[i][j]; ++k) {
                
                DbAlignment* dbAlignment = dbAlignmentsMpi[i][j][k];
                int queryIdx = dbAlignmentGetQueryIdx(dbAlignment);
                
                data[queryIdx].push_back(dbAlignment);
            }
        }
    }
    
    size_t size;
    
    int dbAlignmentsLen = (int) data.size();

    size = dbAlignmentsLen * sizeof(DbAlignment**);
    DbAlignment*** dbAlignments = (DbAlignment***) malloc(size); 
    
    size = dbAlignmentsLen * sizeof(int);
    int* dbAlignmentsLens = (int*) malloc(size);
    
    int i;
    map<int, vector<DbAlignment*> >::iterator it;
    for (it = data.begin(), i = 0; it != data.end(); ++it, ++i) {
    
        sort(it->second.begin(), it->second.end(), dbAlignmentCmp);

        if (maxAlignments < 0) {
            dbAlignmentsLens[i] = (int) it->second.size();
        } else {
            dbAlignmentsLens[i] = min(maxAlignments, (int) it->second.size());
        }
        
        size = dbAlignmentsLens[i] * sizeof(DbAlignment*);
        dbAlignments[i] = (DbAlignment**) malloc(size);
        
        for (int j = 0; j < dbAlignmentsLens[i]; ++j) {
            dbAlignments[i][j] = it->second[j];
        }
    }
    
    *dbAlignments_ = dbAlignments;
    *dbAlignmentsLens_ = dbAlignmentsLens;
    *dbAlignmentsLen_ = dbAlignmentsLen;
}

//******************************************************************************

//******************************************************************************
// PRIVATE

static bool dbAlignmentCmp(DbAlignment* a, DbAlignment* b) {

    double va = dbAlignmentGetValue(a);
    double vb = dbAlignmentGetValue(b);

    return va < vb;
}

//******************************************************************************
