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

#ifndef __SW_SHARP_THREADH__
#define __SW_SHARP_THREADH__

#ifdef _WIN32
#include <windows.h>
#else 
#include <pthread.h>
#include <semaphore.h>
#endif

#ifdef __cplusplus 
extern "C" {
#endif

#ifdef _WIN32
typedef CRITICAL_SECTION Mutex;
typedef HANDLE Semaphore;
typedef HANDLE Thread;
#else 
typedef pthread_mutex_t Mutex;
typedef sem_t Semaphore;
typedef pthread_t Thread;
#endif

extern void mutexCreate(Mutex* mutex);
extern void mutexDelete(Mutex* mutex);
extern void mutexLock(Mutex* mutex);
extern void mutexUnlock(Mutex* mutex);

extern void semaphoreCreate(Semaphore* semaphore, unsigned int value);
extern void semaphoreDelete(Semaphore* semaphore);
extern void semaphorePost(Semaphore* semaphore);
extern int semaphoreValue(Semaphore* semaphore);
extern void semaphoreWait(Semaphore* semaphore);

extern void threadCancel(Thread thread);
extern void threadCreate(Thread* thread, void* (*ruotine)(void*), void* args);
extern void threadCurrent(Thread* thread);
extern void threadJoin(Thread thread);
extern void threadSleep(unsigned int ms);
#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_THREADH__
