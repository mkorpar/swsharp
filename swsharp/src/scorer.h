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

/*!
@brief Scorer object used for alignment scoring.

Scorer is organized as a similarity table with additional gap penalties. Affine 
gap penalty model is used. Scorer codes are defined from 0 to #SCORER_MAX_CODE, 
which coresponds to the letters of the english alphabet. In other words code 0 
represents 'A' and 'a' letters, code 2 'B' and 'b' letters and so on. Scorer 
uses a similarity table where rows and columns correspong to input codes.  
*/
typedef struct Scorer Scorer;

/*!
@brief Scorer constructor.

Construct scorer object with a given name, similarity scores table and affine
gap model penalties.

@param name scorer name, copy is made
@param scores similarity table
@param gapOpen gap open penalty given as a positive integer 
@param gapExtend gap extend penalty given as a positive integer 

@return scorer object
*/
extern Scorer* scorerCreate(const char* name, int* scores, char maxCode, 
    int gapOpen, int gapExtend);
    
/*!
@brief Scorer destructor.

@param scorer scorer object
*/
extern void scorerDelete(Scorer* scorer);

/*!
@brief Gap extend penalty getter.

Gap extend penalty is defined as a positive integer.

@param scorer scorer object

@return gap extend penalty
*/
extern int scorerGetGapExtend(Scorer* scorer);

/*!
@brief Gap open penalty getter.

Gap open penalty is defined as a positive integer.

@param scorer scorer object

@return gap open penalty
*/
extern int scorerGetGapOpen(Scorer* scorer);


extern char scorerGetMaxCode(Scorer* scorer);

/*!
@brief Max score getter.

Max score is defined as the highest score two codes can be scored.

@param scorer scorer object

@return max score
*/
extern int scorerGetMaxScore(Scorer* scorer);

/*!
@brief Name getter.

Scorer name usually coresponds to similarity matrix names.

@param scorer scorer object

@return name
*/
extern const char* scorerGetName(Scorer* scorer);

/*!
@brief Scalar getter.

Getter for scalar property. Scorer is scalar if the similarity matrix can be 
reduced to match, mismatch scorer. In other words scorer is scalar if every two
equal codes are scored equaly and every two unequal codes are scored equaly.

@param scorer scorer object

@return 1 if scorer if scalar 0 otherwise
*/
extern int scorerIsScalar(Scorer* scorer);

/*!
@brief Scores two codes.

Given scorer scores two given codes. Both codes should be greater or equal to 0
and less than #SCORER_MAX_CODE. 

@param scorer scorer object
@param a first code
@param b second code

@return similarity score of a and b
*/
extern int scorerScore(Scorer* scorer, char a, char b);

/*!
@brief Scorer deserialization method.

Method deserializes scorer object from a byte buffer.

@param bytes byte buffer

@return scorer object
*/
extern Scorer* scorerDeserialize(char* bytes);

/*!
@brief Scorer serialization method.

Method serializes scorer object to a byte buffer.

@param bytes output byte buffer
@param bytesLen output byte buffer length
@param scorer scorer object
*/
extern void scorerSerialize(char** bytes, int* bytesLen, Scorer* scorer);

extern char scorerDecode(char c);

extern char scorerEncode(char c);

#ifdef __cplusplus 
}
#endif
#endif // __SWSHARP_SCORERH__
