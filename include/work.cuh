#pragma once
#include <cstdint>
#include "database.cuh"

#define CAPACITY 1 * 1024


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
    // 'top' is the index of the next free slot (i.e., valid items are in indices 0..top-1).
    unsigned int top;
    unsigned int active;

    // Initialize the stack on the device (and host).
    __device__ __host__ void init() {
        top = 0;
        active = 0;
    }
    
    // Device push using a CAS loop.
    __device__ bool push(WorkItem item) {
        unsigned int oldTop, newTop;
        // Loop until we successfully update 'top'
        do {
            oldTop = top;
            // Check if the stack is full.
            if (oldTop >= CAPACITY) {
                return false;
            }
            newTop = oldTop + 1;
        } while (atomicCAS(&top, oldTop, newTop) != oldTop);
        
        // Now we have reserved slot 'oldTop'; store the item.
        items[oldTop] = item;
        atomicAdd(&active, 1);
        // // Increment 'active' using a CAS loop.
        // unsigned int oldActive, newActive;
        // do {
        //     oldActive = active;
        //     newActive = oldActive + 1;
        // } while (atomicCAS(&active, oldActive, newActive) != oldActive);
        
        return true;
    }

    // Device pop using a CAS loop.
    __device__ bool pop(WorkItem *item) {
        unsigned int oldTop, newTop;
        // Loop until we successfully update 'top'
        do {
            oldTop = top;
            // If the stack is empty, return false.
            if (oldTop == 0) {
                return false;
            }
            newTop = oldTop - 1;
        } while (atomicCAS(&top, oldTop, newTop) != oldTop);
        
        // Retrieve the item from the reserved slot.
        *item = items[newTop];
        return true;
    }

    // When a task is finished, decrement the active counter using a CAS loop.
    __device__ void finish_task() {
        atomicSub(&active, 1);
        // unsigned int oldActive, newActive;
        // do {
        //     oldActive = active;
        //     newActive = oldActive - 1;
        // } while (atomicCAS(&active, oldActive, newActive) != oldActive);
    }
    
    // Host initialization helper.
    __host__ void init_host() {
        top = 0;
        active = 0;
    }
};
