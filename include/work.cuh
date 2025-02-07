#pragma once
#include <cstdint>
#include "database.cuh"

#define CAPACITY 256 * 1024


struct WorkItem {

    int *pattern;
    int utility;

    Database *db;

    int max_item;

    int primary;

    int *work_done;
    int work_count;

};



// struct AtomicWorkStack {
//     WorkItem items[CAPACITY];
//     // 'top' is the index of the next free slot (i.e., valid items are in indices 0..top-1).
//     unsigned int top;
//     // A counter for the number of work items pushed.
//     unsigned int active;
// };

// // Initialize the stack (to be called from a kernel or __device__ function).
// __device__ void stack_init(AtomicWorkStack *s) {
//     s->top = 0;
//     s->active = 0;
// }

// // Optimized push using atomicAdd
// __device__ bool stack_push(AtomicWorkStack *s, WorkItem item) {
//     // Reserve a slot by incrementing top.
//     unsigned int idx = atomicAdd(&s->top, 1);
//     if (idx >= CAPACITY) {
//         // Exceeded capacity – roll back.
//         atomicSub(&s->top, 1);
//         return false;
//     }
//     // Write the item into the reserved slot.
//     s->items[idx] = item;
//     // Optionally update active count.
//     atomicAdd(&s->active, 1);
//     // Ensure that the write to items[idx] is visible.
//     __threadfence_system();
//     return true;
// }

// // Optimized pop using atomicSub
// __device__ bool stack_pop(AtomicWorkStack *s, WorkItem *out) {
//     // Decrement top.
//     unsigned int idx = atomicSub(&s->top, 1);
//     if (idx == 0) {
//         // The stack was empty – roll back the decrement.
//         atomicAdd(&s->top, 1);
//         return false;
//     }
//     // idx was the old value; the valid item is at index idx-1.
//     *out = s->items[idx - 1];
//     return true;
// }

// // Returns the current number of active work items.
// __device__ unsigned int stack_get_work_count(AtomicWorkStack *s) {
//     // Use an atomicAdd of zero to read the active counter.
//     // __threadfence_system();
//     return atomicAdd(&s->active, 0);
// }


// Assume WorkItem and CAPACITY are defined appropriately.
struct AtomicWorkStack {
    WorkItem items[CAPACITY];
    // 'top' is the index of the next free slot (i.e., valid items are in indices 0..top-1).
    unsigned int top;
    // A counter that can track the current number of active work items.
    unsigned int active;
};

// Initialize the stack (to be called from a kernel or __device__ function).
__device__ void stack_init(AtomicWorkStack *s) {
    s->top = 0;
    s->active = 0;
}

// Robust push using an atomicCAS loop.
__device__ bool stack_push(AtomicWorkStack *s, WorkItem item) {
    unsigned int oldTop, newTop;
    // Loop until we can atomically increment 'top'
    do {
        oldTop = s->top;
        if (oldTop >= CAPACITY) {
            // Stack is full.
            return false;
        }
        newTop = oldTop + 1;
        // Attempt to reserve a slot.
    } while (atomicCAS(&s->top, oldTop, newTop) != oldTop);
    
    // We have now reserved slot "oldTop" exclusively.
    s->items[oldTop] = item;
    // Make sure the write to items[oldTop] is visible before we update the active counter.
    __threadfence(); 
    atomicAdd(&s->active, 1);
    
    return true;
}

// Robust pop using an atomicCAS loop.
__device__ bool stack_pop(AtomicWorkStack *s, WorkItem *out) {
    unsigned int oldTop, newTop;
    // Loop until we can successfully decrement 'top'
    do {
        oldTop = s->top;
        if (oldTop == 0) {
            // Stack is empty.
            return false;
        }
        newTop = oldTop - 1;
    } while (atomicCAS(&s->top, oldTop, newTop) != oldTop);
    
    // Now the top element is at index newTop.
    *out = s->items[newTop];
    // Ensure that the read of items[newTop] is complete.
    __threadfence();
    // atomicSub(&s->active, 1);
    
    return true;
}

// Returns the current number of active work items.
// (Since 'top' is updated atomically, it holds the current count.)
__device__ unsigned int stack_get_work_count(AtomicWorkStack *s) {
    // Reading 'top' is sufficient because push/pop update it atomically.
    return s->top;
}