#pragma once
#include <cstdint>
#include "database.cuh"

#define CAPACITY (1024 * 1024)


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
    volatile unsigned int top;    // Points to the next free slot
    volatile unsigned int active; // Keeps track of active tasks
    int lock;                     // 0 = unlocked, 1 = locked

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

    // Device-side spinlock using atomicCAS
    __device__ void acquire_lock() {
        while (atomicCAS(&lock, 0, 1) != 0) {
            // Busy-wait
        }
    }

    __device__ void release_lock() {
        atomicExch(&lock, 0);
    }

    // Push a work item onto the stack (LIFO)
    __device__ bool push(WorkItem item) {
        bool success = false;
        acquire_lock();
        if (top < CAPACITY) {  // Check if stack is not full
            items[top] = item;
            top = top + 1;
            __threadfence();  // Ensure memory ordering
            atomicAdd((unsigned int *)&active, 1);
            success = true;
        }
        release_lock();
        return success;
    }

    // Pop a work item from the stack (LIFO)
    __device__ bool pop(WorkItem *item) {
        bool success = false;
        acquire_lock();
        if (top > 0) {  // Check if stack is not empty
            top = top - 1;
            *item = items[top];
            __threadfence();  // Ensure memory ordering
            success = true;
        }
        release_lock();
        return success;
    }

    __host__ bool host_pop(WorkItem *item) {
        bool success = false;
        if (top > 0) {  // Check if stack is not empty
            top = top - 1;
            *item = items[top];
            success = true;
        }
        return success;
    }

    __device__ int get_active() {
        return active;
    }

    // Mark a task as finished (decrement 'active')
    __device__ void finish_task() {
        atomicSub((unsigned int *)&active, 1);
    }
};
