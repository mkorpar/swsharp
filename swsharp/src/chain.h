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

#ifndef __SW_SHARP_CHAINH__
#define __SW_SHARP_CHAINH__

#ifdef __cplusplus 
extern "C" {
#endif

typedef struct Chain Chain;

extern Chain* chainCreate(char* name, int nameLen, char* string, int stringLen);

extern void chainDelete(Chain* chain);

extern char chainGetChar(Chain* chain, int index);
extern char chainGetCode(Chain* chain, int index);
extern int chainGetLength(Chain* chain);
extern const char* chainGetName(Chain* chain);

extern Chain* chainCreateView(Chain* chain, int start, int end, int reverse);

extern void chainCopyCodes(Chain* chain, char* dest);

extern Chain* chainDeserialize(char* bytes);

extern void chainSerialize(char** bytes, int* bytesLen, Chain* chain);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_CHAINH__
