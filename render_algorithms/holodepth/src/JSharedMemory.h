#ifndef _JSHAREDMEMORY_H_
#define _JSHAREDMEMORY_H_
/*
 *  JSharedMemory.h
 *  SharedMemoryTest
 *
 *  Created by James Barabas on 2/28/10.
 *  Copyright 2010 MIT Med(ia Lab. All rights reserved.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <ctype.h>
#include <string.h>
#include <sys/sem.h>

#ifndef __APPLE__
#ifndef semun
 union semun {        
	int val;                        /* value for SETVAL */
	struct semid_ds *buf;           /* buffer for IPC_STAT & IPC_SET */
	unsigned short *array;          /* array for GETALL & SETALL */
	struct seminfo *__buf;          /* buffer for IPC_INFO */
	void *__pad;
 };  
#endif
#endif


#define SEM_RESOURCE_MAX        1       /* Initial value of all semaphores */
#ifndef SEMMSL
#define SEMMSL 32
#endif


class JSharedMemory
{
	int m_memsize;
	key_t m_key;
	int   m_shmid;
	 char  *m_segptr;
	int m_semset;
	int m_semcount;
	
public:
	JSharedMemory(int memsiz, int key = 6853);
	
	//try to lock and write shared memory. Fail if can't get a lock after a bunch of tries.
	bool write(void *text);
	
	//just print contents of shared segment to stdout
	void print(); 
	
	//get copy of contents with no locking
	inline void getDataCopyUnsafe( void* toPtr) { memcpy(toPtr, m_segptr, m_memsize); }

	//try once to lock shared memory, if available, copy & unlock again
	bool getDataCopyIfUnlocked(void* toPtr);
	bool getDataCopy(void* toPtr);

	//dump shared memory
	void remove();
	
	//change permissions for shared memory
	void changemode(char *mode) ;
	
	void testsem()
	{
		printf("remove:\n");
		removesem();
		printf("open:\n");
		opensem();
		printf("create:\n");
		createsem(2);
		printf("get:\n");
		dispval(0);
		printf("lock:\n");
		locksem(0);
		printf("get:\n");
		dispval(0);
		printf("lock:\n");
		locksem(0);
		printf("get:\n");
		dispval(0);
		printf("unlock:\n");
		unlocksem(0);
		printf("get:\n");
		dispval(0);
		printf("unlock:\n");
		unlocksem(0);
		printf("get:\n");
		dispval(0);
		printf("remove:\n");
		removesem();
	}

	//get pointer to shared memory segment. No locking. Bad idea.
	inline  void* getptr() { return m_segptr; }
private:

	
	//returns false if semaphore does not exist
	bool opensem()
	{
        /* Open the semaphore set - do not create! */
		int theset;
        if((theset = semget(m_key+1, 0, 0666)) == -1) 
        {
			printf("Semaphore set does not exist!\n");
			perror("");
			return false;
        }
		m_semset = theset;
		printf("opened semaphore %d\n", m_semset);
		return true;
		
	}
	
	bool createsem(int members)
	{
		m_semcount = members;
        int cntr;
        union semun semopts;
		
        if(members > SEMMSL) {
			printf("Sorry, max number of semaphores in a set is %d\n",
				   SEMMSL);
			return false;
        }
		
        printf("Attempting to create new semaphore set with %d members\n",
			   members);
		int theset;
        if((theset = semget(m_key+1, members, IPC_CREAT|IPC_EXCL|0666))
		   == -1) 
        {
			fprintf(stderr, "Semaphore set already exists!\n");
			return false;
        }
		m_semset = theset;
        semopts.val = SEM_RESOURCE_MAX;
        
        /* Initialize all members (could be done with SETALL) */        
        for(cntr=0; cntr<members; cntr++)
			semctl(m_semset, cntr, SETVAL, semopts);
		return true;
	}
	
	bool locksem(int member)
	{
        struct sembuf sem_lock={ 0, -1, IPC_NOWAIT};
		
        if( member<0 || member>(get_member_count()-1))
        {
			fprintf(stderr, "semaphore member %d out of range\n", member);
			return false;
        }
		
        /* Attempt to lock the semaphore set */
        if(!getval(member))
        {
			//fprintf(stderr, "Semaphore resources exhausted (no lock)!\n");
			return false;
        }
        
        sem_lock.sem_num = member;
        
        if((semop(m_semset, &sem_lock, 1)) == -1)
        {
			//perror("lock");
			//fprintf(stderr, "Lock not available\n");
			return false;
        }
        else
		{
			//printf("Semaphore resources decremented by one (locked)\n");
		}
        //dispval(member);
		return true;
	}
	
	bool unlocksem(int member)
	{
        struct sembuf sem_unlock={ member, 1, IPC_NOWAIT};
        int semval;
		
        if( member<0 || member>(get_member_count()-1))
        {
			fprintf(stderr, "semaphore member %d out of range\n", member);
			return false;
        }
		
        /* Is the semaphore set locked? */
        semval = getval(member);
        if(semval == SEM_RESOURCE_MAX) {
			fprintf(stderr, "Semaphore not unlocked!\n");
			return false;
        }
		
        sem_unlock.sem_num = member;
		
        /* Attempt to lock the semaphore set */
		if((semop(m_semset, &sem_unlock, 1)) == -1)
        {
			fprintf(stderr, "Unlock failed\n");
			return false;
        }
        else
		{
			//printf("Semaphore resources incremented by one (unlocked)\n");
		}
        //dispval(member);
		return true;
	}
	
	void removesem()
	{
        int ret = semctl(m_semset, 0, IPC_RMID, 0);
        printf("Semaphore removed (status %d)\n",ret);
	}
	
	
	unsigned short get_member_count()
	{
		return m_semcount;

        //union semun semopts;
        //struct semid_ds mysemds;
		
        //semopts.buf = &mysemds;
		
		//int rc = semctl(m_semset, 0, IPC_STAT, semopts);
		
		//if (rc == -1) {
		//	perror("semctl failed");
		//	exit(1);
        //}
		
        // Return number of members in the semaphore set /
        //return(semopts.buf->sem_nsems);
	}
	
	int getval(int member)
	{
        int semval;
		
        semval = semctl(m_semset, member, GETVAL, 0);
        return(semval);
	}
	
	void changesemmode(char *mode)
	{
        int rc;
        union semun semopts;
        struct semid_ds mysemds;
		
        /* Get current values for internal data structure */
        semopts.buf = &mysemds;
		
        rc = semctl(m_semset, 0, IPC_STAT, semopts);
		
        if (rc == -1) {
			perror("semctl");
			exit(1);
        }
		
        printf("Old permissions were %o\n", semopts.buf->sem_perm.mode);
		
        /* Change the permissions on the semaphore */
        sscanf(mode, "%ho", &semopts.buf->sem_perm.mode);
		
        /* Update the internal data structure */
        semctl(m_semset, 0, IPC_SET, semopts);
		
        printf("Updated...\n");
		
	}
	
	void dispval(int member)
	{
        int semval;
		
        semval = semctl(m_semset, member, GETVAL, 0);
        printf("semval for member %d is %d\n", member, semval);
	}
	
	
	
};
#endif
