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
    volatile unsigned int front;
    volatile unsigned int rear;
    volatile unsigned int active;  // Keeps track of active tasks
    int lock;  // 0 = unlocked, 1 = locked

    __device__ __host__ void init() {
        front = 0;
        rear = 0;
        active = 0;
        lock = 0;
        memset(items, 0, sizeof(items));
    }

    __host__ void init_host() {
        front = 0;
        rear = 0;
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

    // Enqueue a work item (FIFO)
    __device__ bool push(WorkItem item) {
        bool success = false;
        acquire_lock();
        if ((rear + 1) % CAPACITY != front) {  // Check if queue is not full
            items[rear] = item;
            rear = (rear + 1) % CAPACITY;
            atomicAdd((unsigned int *)&active, 1);
            success = true;
        }
        release_lock();
        return success;
    }

    // Dequeue a work item
    __device__ bool pop(WorkItem *item) {
        bool success = false;
        acquire_lock();
        if (front != rear) {  // Check if queue is not empty
            *item = items[front];
            front = (front + 1) % CAPACITY;
            __threadfence();
            success = true;
        }
        release_lock();
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
