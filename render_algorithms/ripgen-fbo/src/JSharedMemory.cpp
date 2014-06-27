/*
 *  JSharedMemory.cpp
 *  SharedMemoryTest
 *
 *  Created by James Barabas on 2/28/10.
 *  Copyright 2010 MIT Media Lab. All rights reserved.
 *
 */

#include "JSharedMemory.h"

#include <errno.h>

JSharedMemory::JSharedMemory(int memsiz, int key)
{
	m_memsize = memsiz;
	m_key = key;
	
	/* Create unique key via call to ftok() */
	//key = ftok(".", 'S');
	
	/* Open the shared memory segment - create if necessary */

    /* Mode bits */
    //#define	IPC_CREAT	001000		/* Create entry if key does not exist */
    //#define	IPC_EXCL	002000		/* Fail if key exists */
    //#define	IPC_NOWAIT	004000		/* Error if request must wait */


	m_shmid = shmget(m_key, m_memsize, IPC_CREAT|IPC_EXCL|0666);
    if(m_shmid == -1) 
	{
		printf("Shared memory segment exists - opening as client [shmget(%X,%d,%o)], err: %d\n" ,m_key,m_memsize,IPC_CREAT|IPC_EXCL|0666, 1*(errno) );
		
        if(errno == EINVAL) printf("[EINVAL]\tNo shared memory segment is to be created, and a \n\
                                     shared memory segment exists for key, but the size of \n\
                                     the segment associated with it is less than size, \n\
                                     which is non-zero.\n");
        perror("in opening shared memory segment: ");
        
		/* Segment probably already exists - try as a client */
		if((m_shmid = shmget(m_key, m_memsize, 0)) == -1) 
		{
            printf("Got -1 from shmget(%X,%d,0), err: %d\n",m_key,m_memsize, errno);
            if(errno == ENOENT) printf("[ENOENT]\tIPC_CREAT was not set in shmflg and no shared memory segment \n\
                                       associated with key was found.\n ");
            
            perror("shmget failed. Exiting. You may need to increase shared memory using sysctl.\n");
			exit(1);
		}
	}
	else
	{
		printf("Creating new shared memory segment\n");
	}
	
	/* Attach (map) the shared memory segment into the current process */
	if((m_segptr = (char *)shmat(m_shmid, 0, 0)) == (char *)-1)
	{
		perror("shmat failed. Exiting.");
		exit(1);
	}
	
	//removesem();
	createsem(1);
	opensem();

}

bool JSharedMemory::getDataCopyIfUnlocked(void* toPtr) 
{ 
	if(!locksem(0)) return false;
	memcpy(toPtr, m_segptr, m_memsize); 
	unlocksem(0);
	return true;
}

bool JSharedMemory::getDataCopy(void* toPtr) 
{ 
	int maxtries = 10000;
	int tried = 0;
	for(int tried=0;tried < maxtries;tried++)
	{
		if(locksem(0)) break;
	}
	if(tried>=maxtries-1)
	{
		printf("shared memory read failed -- could not get lock after waiting. Data not copied\n");
		return false;
	}
	memcpy(toPtr, m_segptr, m_memsize); 
	unlocksem(0);
	return true;
}


bool JSharedMemory::write(void *fromPtr)
{
	int maxtries = 10000;
	int tried = 0;
	for(tried=0;tried<maxtries;tried++)
	{
		if(locksem(0))
		{
			break;
		}
	} //loop until semaphore locks
	
	if(tried>=maxtries-1)
	{
		printf("shared memory write failed -- could not get lock after waiting. Data not written.\n");
		return false;
	}
	
	memcpy(m_segptr,fromPtr, m_memsize);
	unlocksem(0);
	//printf("Done...\n");
	return true;
}

void JSharedMemory::print()
{
	char buf [1024];
	if(getDataCopyIfUnlocked(buf))
	{
		printf("m_segptr: %s\n", buf);
	}
}

void JSharedMemory::remove()
{
	shmctl(m_shmid, IPC_RMID, 0);
	printf("Shared memory segment marked for deletion\n");
}

void JSharedMemory::changemode(char *mode) 
{
	struct shmid_ds myshmds;
	
	/* Get current values for internal data structure */
	shmctl(m_shmid, IPC_STAT, &myshmds);
	
	/* Display old permissions */
	printf("Old permissions were: %o\n", myshmds.shm_perm.mode);
	
	/* Convert and load the mode */ 
	sscanf(mode, "%o", (unsigned int*)&myshmds.shm_perm.mode);
	
	/* Update the mode */
	shmctl(m_shmid, IPC_SET, &myshmds);
	
	printf("New permissions are : %o\n", myshmds.shm_perm.mode);
}
