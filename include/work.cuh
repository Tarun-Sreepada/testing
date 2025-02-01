#pragma once
#include <cstdint>

#define CAPACITY 8192

struct Item {
    int key;
    int util;
} __attribute__((packed)); // Ensures no padding


struct WorkItem {
    int *pattern;
    Item *items;
    int num_items;
    int *start;
    int *end;
    int *utility;
    int num_transactions;
    int primary_item;
    int max_item;

    int *work_done;
    int work_count;

    int bytes;
    void *base_ptr;

};


struct ProjectionMemory{
    int *n_pattern;
    Item *n_items;
    int *n_start;
    int *n_end;
    int *n_utility;
    Item *tran_hash;
    Item *local_util;
    Item *subtree_util;
    int *work_done;
    int bytes_to_alloc;
    void *base_ptr;
} ;


typedef struct {
    int item_counter;
    int transaction_counter;
    int pattern_utility;
} ProjectionResult;


typedef struct {
    int new_item_counter;
    int new_transaction_counter;
} MergeResult;



// The atomic work stack structure stored in device memory.
struct AtomicWorkStack {
    WorkItem items[CAPACITY];
    // 'top' is the index of the next free slot (i.e., valid items are in indices 0..top-1).
    unsigned int top;
    // A counter for the number of work items pushed.
    unsigned int active;
};

// Initialize the stack (to be called from a kernel or a __device__ function).
__device__ void stack_init(AtomicWorkStack *s) {
    s->top = 0;
    s->active = 0;
}

// Push (thread-safe) – equivalent to DFS "enqueue".
// Returns true if the push succeeds, false if the stack is full.
__device__ bool stack_push(AtomicWorkStack *s, WorkItem item) {
    while (true) {
        // Read the current top value.
        unsigned int curTop = s->top;
        // If the stack is full, return false.
        if (curTop >= CAPACITY) {
            return false;
        }
        // Attempt to claim the slot at index 'curTop' by performing an atomic compare-and-swap.
        // atomicCAS returns the old value at s->top.
        unsigned int prev = atomicCAS(&(s->top), curTop, curTop + 1);
        if (prev == curTop) {
            // We succeeded in incrementing the top.
            s->items[curTop] = item;
            // Atomically increment active.
            atomicAdd(&(s->active), 1);
            __threadfence();
            return true;
        }
        // Otherwise, another thread updated s->top, so try again.
    }
}

// Pop (thread-safe) – equivalent to DFS "dequeue".
// Returns true and writes the popped item to *out if successful; returns false if the stack is empty.
__device__ bool stack_pop(AtomicWorkStack *s, WorkItem *out) {
    while (true) {
        // Read the current top value.
        unsigned int curTop = s->top;
        // If the stack is empty, return false.
        if (curTop == 0) {
            return false;
        }
        unsigned int newTop = curTop - 1;
        // Attempt to claim the top item by atomically decrementing s->top.
        unsigned int prev = atomicCAS(&(s->top), curTop, newTop);
        if (prev == curTop) {
            // We successfully decremented the top; retrieve the item.
            *out = s->items[newTop];
            return true;
        }
        // Otherwise, retry.
    }
}

// Get the current work_count.
__device__ unsigned int stack_get_work_count(AtomicWorkStack *s) {
    // Since work_count is updated atomically via atomicAdd, a plain read is acceptable.
    // return s->work_count;
    return atomicAdd(&(s->active), 0);
}
