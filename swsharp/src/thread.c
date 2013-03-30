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

#include <limits.h>

#ifdef _WIN32
#include <windows.h>
#else 
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#endif

#include "thread.h"

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// MUTEX

extern void mutexCreate(Mutex* mutex) {
#ifdef _WIN32
    InitializeCriticalSection(mutex);
#else 
    // do nothing
#endif
}

extern void mutexDelete(Mutex* mutex) {
#ifdef _WIN32
    DeleteCriticalSection(mutex);
#else 
    // do nothing
#endif
}

extern void mutexLock(Mutex* mutex) {
#ifdef _WIN32
    EnterCriticalSection(mutex);
#else 
    pthread_mutex_lock(mutex);
#endif
}

extern void mutexUnlock(Mutex* mutex) {
#ifdef _WIN32
    LeaveCriticalSection(mutex);
#else 
    pthread_mutex_unlock(mutex);
#endif
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SEMAPHORES

extern void semaphoreCreate(Semaphore* semaphore, unsigned int value) {
#ifdef _WIN32
    *semaphore = CreateSemaphore(NULL, value, INT_MAX, NULL);
#else 
    sem_init(semaphore, 0, value);
#endif
}

extern void semaphoreDelete(Semaphore* semaphore) {
#ifdef _WIN32
    CloseHandle(*semaphore);
#else 
    sem_destroy(semaphore);
#endif
}

extern void semaphorePost(Semaphore* semaphore) {
#ifdef _WIN32
    ReleaseSemaphore(*semaphore, 1, NULL);
#else 
    sem_post(semaphore);
#endif
}

extern int semaphoreValue(Semaphore* semaphore) {
#ifdef _WIN32
    long value;
    ReleaseSemaphore(*semaphore, 1, &value);
    WaitForSingleObject(*semaphore, INFINITE);
    return (int) value;
#else 
    int value; 
    sem_getvalue(semaphore, &value); 
    return value;
#endif
}

extern void semaphoreWait(Semaphore* semaphore) {
#ifdef _WIN32
    WaitForSingleObject(*semaphore, INFINITE);
#else 
    sem_wait(semaphore);
#endif
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// THREADS

extern void threadCancel(Thread thread) {
#ifdef _WIN32
    TerminateThread(thread, 0);
#else 
    pthread_cancel(thread);
    pthread_join(thread, NULL);
#endif
}

extern void threadCreate(Thread* thread, void* (*fun)(void*), void* args) {
#ifdef _WIN32
    *thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) fun, args, 0, NULL);
#else 
    pthread_create(thread, NULL, fun, args);
#endif
}

extern void threadCurrent(Thread* thread) {
#ifdef _WIN32
    *thread = GetCurrentThread();
#else 
    *thread = pthread_self();
#endif
}

extern void threadJoin(Thread thread) {
#ifdef _WIN32
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else 
    pthread_join(thread, NULL);
#endif
}

extern void threadSleep(unsigned int ms) {
#ifdef _WIN32
    Sleep(ms);
#else 
    usleep(ms * 1000);
#endif
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************
