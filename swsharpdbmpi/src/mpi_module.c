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

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "swsharp/swsharp.h"

#include "mpi_module.h"

//******************************************************************************
// PUBLIC

extern void sendMpiData(DbAlignment*** dbAlignments, int* dbAlignmentsLens, 
    int dbAlignmentsLen, int node);
    
extern void recieveMpiData(DbAlignment**** dbAlignments, int** dbAlignmentsLens, 
    int* dbAlignmentsLen, Chain** queries, Chain** database, Scorer* scorer, 
    int node);
    
//******************************************************************************

//******************************************************************************
// PRIVATE

static DbAlignment* dbAlignmentFromBytes(char* bytes, Chain** queries, 
    Chain** database, Scorer* scorer);

static void dbAlignmentToBytes(char** bytes, size_t* size, 
    DbAlignment* dbAlignment);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void recieveMpiData(DbAlignment**** dbAlignments_, 
    int** dbAlignmentsLens_, int* dbAlignmentsLen_, Chain** queries, 
    Chain** database, Scorer* scorer, int node) {

    int i, j;
    MPI_Status status;
    
    size_t size;
    MPI_Recv(&size, sizeof(size), MPI_CHAR, node, 1, MPI_COMM_WORLD, &status);

    char* buffer = (char*) malloc(size);
    MPI_Recv(buffer, size, MPI_CHAR, node, 0, MPI_COMM_WORLD, &status);
      
    size_t ptr = 0;
    
    int dbAlignmentsLen;
    memcpy(&dbAlignmentsLen, buffer + ptr, sizeof(int));
    ptr += sizeof(int);  
    
    size = dbAlignmentsLen * sizeof(DbAlignment**);
    DbAlignment*** dbAlignments = (DbAlignment***) malloc(size);
    
    size = dbAlignmentsLen * sizeof(int*);
    int* dbAlignmentsLens = (int*) malloc(size);
    
    for (i = 0; i < dbAlignmentsLen; ++i) {
    
        memcpy(&(dbAlignmentsLens[i]), buffer + ptr, sizeof(int));
        ptr += sizeof(int);
        
        size = dbAlignmentsLens[i] * sizeof(DbAlignment*);
        dbAlignments[i] = (DbAlignment**) malloc(size);
        
        for (j = 0; j < dbAlignmentsLens[i]; ++j) {
        
            size_t bytesSize;
            memcpy(&bytesSize, buffer + ptr, sizeof(size_t));
            ptr += sizeof(size_t);
            
            dbAlignments[i][j] = dbAlignmentFromBytes(buffer + ptr, queries, 
                database, scorer);
            ptr += bytesSize;
        }
    }
    
    *dbAlignments_ = dbAlignments;
    *dbAlignmentsLens_ = dbAlignmentsLens;
    *dbAlignmentsLen_ = dbAlignmentsLen; 
}

extern void sendMpiData(DbAlignment*** dbAlignments, int* dbAlignmentsLens, 
    int dbAlignmentsLen, int node) {
    
    int i, j;
    
    const int bufferStep = 4096;
    
    size_t bufferSize = bufferStep;
    size_t realSize = 0;
    char* buffer = (char*) malloc(bufferSize);

    size_t ptr = 0;

    memcpy(buffer + ptr, &dbAlignmentsLen, sizeof(int));
    ptr += sizeof(int);
    realSize += sizeof(int);

    for (i = 0; i < dbAlignmentsLen; ++i) {
        
        realSize += sizeof(int); // dbAlignmentsLens[i]
        
        if (realSize >= bufferSize) {
            bufferSize += (realSize - bufferSize) + bufferStep;
            buffer = (char*) realloc(buffer, bufferSize);
        }

        memcpy(buffer + ptr, &(dbAlignmentsLens[i]), sizeof(int));
        ptr += sizeof(int);

        for (j = 0; j < dbAlignmentsLens[i]; ++j) {
        
            size_t bytesSize;
            char* bytes;
            
            dbAlignmentToBytes(&bytes, &bytesSize, dbAlignments[i][j]);
            
            realSize += sizeof(size_t);
            realSize += bytesSize;
            
            if (realSize >= bufferSize) {
                bufferSize += (realSize - bufferSize) + bufferStep;
                buffer = (char*) realloc(buffer, bufferSize);
            }
            
            memcpy(buffer + ptr, &bytesSize, sizeof(size_t));
            ptr += sizeof(size_t);
            
            memcpy(buffer + ptr, bytes, bytesSize);
            ptr += bytesSize;
        }
    }
    
    MPI_Send(&realSize, sizeof(size_t), MPI_CHAR, node, 1, MPI_COMM_WORLD);
    MPI_Send(buffer, realSize, MPI_CHAR, node, 0, MPI_COMM_WORLD);
    
    free(buffer);
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// SERIALIZATION

static DbAlignment* dbAlignmentFromBytes(char* bytes, Chain** queries, 
    Chain** database, Scorer* scorer) {

    int ptr = 0;
    
    int queryStart;
    memcpy(&queryStart, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    int queryEnd;
    memcpy(&queryEnd, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int queryIdx;
    memcpy(&queryIdx, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int targetStart;
    memcpy(&targetStart, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int targetEnd;
    memcpy(&targetEnd, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int targetIdx;
    memcpy(&targetIdx, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    int score;
    memcpy(&score, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    double value;
    memcpy(&value, bytes + ptr, sizeof(double));
    ptr += sizeof(double);
    
    int pathLen;
    memcpy(&pathLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    char* path = (char*) malloc(pathLen);
    memcpy(path, bytes + ptr, pathLen);
    
    Chain* query = queries[queryIdx];
    Chain* target = database[targetIdx];

    DbAlignment* dbAlignment = dbAlignmentCreate(query, queryStart, queryEnd,
        queryIdx, target, targetStart, targetEnd, targetIdx, value, score, 
        scorer, path, pathLen);
    
    return dbAlignment;
}

static void dbAlignmentToBytes(char** bytes, size_t* size, 
    DbAlignment* dbAlignment) {
    
    // int 3 query
    // int 3 target
    // int 1 score
    // double 1 value
    // int 1 pathLen
    // char pathLen path
    *size = sizeof(int) * 8 + sizeof(double) + dbAlignmentGetPathLen(dbAlignment);
    *bytes = (char*) malloc(*size);

    int ptr = 0;

    int queryStart = dbAlignmentGetQueryStart(dbAlignment);
    memcpy(*bytes + ptr, &queryStart, sizeof(int));
    ptr += sizeof(int);
    
    int queryEnd = dbAlignmentGetQueryEnd(dbAlignment);
    memcpy(*bytes + ptr, &queryEnd, sizeof(int));
    ptr += sizeof(int);
    
    int queryIdx = dbAlignmentGetQueryIdx(dbAlignment);
    memcpy(*bytes + ptr, &queryIdx, sizeof(int));
    ptr += sizeof(int);
    
    int targetStart = dbAlignmentGetTargetStart(dbAlignment);
    memcpy(*bytes + ptr, &targetStart, sizeof(int));
    ptr += sizeof(int);

    int targetEnd = dbAlignmentGetTargetEnd(dbAlignment);
    memcpy(*bytes + ptr, &targetEnd, sizeof(int));
    ptr += sizeof(int);

    int targetIdx = dbAlignmentGetTargetIdx(dbAlignment);
    memcpy(*bytes + ptr, &targetIdx, sizeof(int));
    ptr += sizeof(int);

    int score = dbAlignmentGetScore(dbAlignment);
    memcpy(*bytes + ptr, &score, sizeof(int));
    ptr += sizeof(int);
    
    double value = dbAlignmentGetValue(dbAlignment);
    memcpy(*bytes + ptr, &value, sizeof(double));
    ptr += sizeof(double);
    
    int pathLen = dbAlignmentGetPathLen(dbAlignment);
    memcpy(*bytes + ptr, &pathLen, sizeof(int));
    ptr += sizeof(int);

    dbAlignmentCopyPath(dbAlignment, *bytes + ptr);
    ptr += pathLen;
}

//------------------------------------------------------------------------------
//******************************************************************************
