#pragma once
#include <cstdint>
#include "database.cuh"

#define CAPACITY (128 * 1024)


struct WorkItem {

    int *pattern;
    int pattern_length;

    int utility;

    Database *db;

    int max_item;

    int primary;

    int *work_done;
    int work_count;

};


struct AtomicWorkStack {
    WorkItem items[CAPACITY];
    volatile unsigned int top;
    volatile unsigned int active;  // volatile so threads always reload its value
    
    // Add a lock variable (0 = unlocked, 1 = locked).
    int lock;

    __device__ __host__ void init() {
        top = 0;
        active = 0;
        lock = 0;
        memset(items, 0, sizeof(items));
    }
    
    __host__ void init_host() {
        top = 0;
        active = 0;
        lock = 0;
    }

    // Simple device-side spinlock using atomicCAS.
    __device__ void acquire_lock() {
        while (atomicCAS(&lock, 0, 1) != 0) {
            // Busy-wait until the lock is acquired.
            // __nanosleep(1000);
        }
    }

    __device__ void release_lock() {
        atomicExch(&lock, 0);
    }
    
    // Push a work item using the lock.
    __device__ bool push(WorkItem item) {
        bool success = false;
        acquire_lock();
        if (top < CAPACITY) {
            items[top] = item;
            top = top + 1;
            success = true;
            // Update the global push counter.
            // Increment 'active'
            atomicAdd((unsigned int *)&active, 1);
        }
        release_lock();
        return success;
    }

    // Pop a work item using the lock.
    __device__ bool pop(WorkItem *item) {
        bool success = false;
        acquire_lock();
        if (top > 0) {
            top = top - 1;
            *item = items[top];
            success = true;
        }
        release_lock();
        return success;
    }

    __device__ int get_active() {
        // __threadfence();  // Optional for consistency.
        return active;
    }

    // Mark a task as finished (decrement 'active').
    __device__ void finish_task() {
        atomicSub((unsigned int *)&active, 1);
    }
};
