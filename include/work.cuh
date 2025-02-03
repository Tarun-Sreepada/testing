#pragma once
#include <cstdint>
#include "database.cuh"

#define CAPACITY 256 * 1024


struct WorkItem {

    int *pattern;

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
    // A counter for the number of work items pushed.
    unsigned int active;
};

// Initialize the stack (to be called from a kernel or __device__ function).
__device__ void stack_init(AtomicWorkStack *s) {
    s->top = 0;
    s->active = 0;
}

// Optimized push using atomicAdd
__device__ bool stack_push(AtomicWorkStack *s, WorkItem item) {
    // Reserve a slot by incrementing top.
    unsigned int idx = atomicAdd(&s->top, 1);
    if (idx >= CAPACITY) {
        // Exceeded capacity – roll back.
        atomicSub(&s->top, 1);
        return false;
    }
    // Write the item into the reserved slot.
    s->items[idx] = item;
    // Optionally update active count.
    atomicAdd(&s->active, 1);
    // Ensure that the write to items[idx] is visible.
    __threadfence_system();
    return true;
}

// Optimized pop using atomicSub
__device__ bool stack_pop(AtomicWorkStack *s, WorkItem *out) {
    // Decrement top.
    unsigned int idx = atomicSub(&s->top, 1);
    if (idx == 0) {
        // The stack was empty – roll back the decrement.
        atomicAdd(&s->top, 1);
        return false;
    }
    // idx was the old value; the valid item is at index idx-1.
    *out = s->items[idx - 1];
    return true;
}

// Returns the current number of active work items.
__device__ unsigned int stack_get_work_count(AtomicWorkStack *s) {
    // Use an atomicAdd of zero to read the active counter.
    // __threadfence_system();
    return atomicAdd(&s->active, 0);
}

__device__ void stack_peek(AtomicWorkStack *s, WorkItem *out) {
    unsigned int idx = s->top;
    if (idx == 0) {
        return;
    }
    *out = s->items[idx - 1];
}





// struct ProjectionMemory{
//     int *n_pattern;
//     Item *n_items;
//     int *n_start;
//     int *n_end;
//     int *n_utility;
//     Item *tran_hash;
//     Item *local_util;
//     Item *subtree_util;
//     int *work_done;
//     int bytes_to_alloc;
//     void *base_ptr;
// } ;


// typedef struct {
//     int item_counter;
//     int transaction_counter;
//     int pattern_utility;
// } ProjectionResult;


// typedef struct {
//     int new_item_counter;
//     int new_transaction_counter;
// } MergeResult;




    // Item *items;
    // int num_items;
    // int *start;
    // int *end;
    // int *utility;
    // int num_transactions;