#pragma once
#include <cstdint>


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
};



template<typename T, int CAPACITY>
struct AtomicWorkQueue {
    T items[CAPACITY];
    unsigned int head;
    unsigned int tail;
    uint32_t work_count;

    // Initialize the queue
    __host__ __device__ void init() {
        head = 0;
        tail = 0;
        work_count = 0;
    }

    // Atomic read of work count
    __device__ uint32_t get_work_count() {
        return atomicAdd(&work_count, 0);  // Atomic read
    }

    __device__ bool enqueue(const T &item) {
        unsigned int curTail, curHead, nextTail;
        bool success = false;
        
        do {
            curTail = atomicAdd(&tail, 0);
            curHead = atomicAdd(&head, 0);
            nextTail = (curTail + 1) % CAPACITY;

            if (nextTail == curHead) return false;  // Full
            
            // Attempt to claim slot
            if (atomicCAS(&tail, curTail, nextTail) == curTail) {
                items[curTail] = item;
                atomicAdd(&work_count, 1);  // Increment ONLY after successful insert
                __threadfence();  // Ensure memory consistency
                success = true;
                break;
            }
        } while (true);

        return success;
    }

    __device__ bool dequeue(T &out) {
        unsigned int curHead, curTail, nextHead;
        bool success = false;

        do {
            curHead = atomicAdd(&head, 0);
            curTail = atomicAdd(&tail, 0);

            if (curHead == curTail) return false;  // Empty
            
            nextHead = (curHead + 1) % CAPACITY;
            
            // Attempt to claim item
            if (atomicCAS(&head, curHead, nextHead) == curHead) {
                out = items[curHead];
                __threadfence();  // Ensure memory consistency
                success = true;
                break;
            }
        } while (true);

        return success;
    }

    // Host-side enqueue (keep original implementation)
    __host__ bool host_enqueue(const T &item) {
        unsigned int curTail = tail;
        unsigned int nextTail = (curTail + 1) % CAPACITY;
        
        if (nextTail == head) return false;
        
        items[curTail] = item;
        tail = nextTail;
        work_count++;  // No atomics needed for host-side
        return true;
    }

    // Host-side dequeue
    __host__ bool host_dequeue(T &out) {
        unsigned int curHead = head;
        unsigned int curTail = tail;

        if (curHead == curTail) return false;
        
        out = items[curHead];
        head = (curHead + 1) % CAPACITY;
        return true;
    }
};

