#include <cstdio>
#include <cuda_runtime.h>

#define CAPACITY (128*1024)
#define INITIAL_TASKS (1 * 1024)
#define NUM_THREADS (1 * 1024)
#define NEW_TASKS_PER_POP 128

// nvcc -o stack stack.cu && compute-sanitizer ./stack

// Global managed counters for push/pop operations.
__managed__ unsigned int pushCount = 0;  // Total successful pushes.
__managed__ unsigned int popCount = 0;   // Total completed (finished) tasks.

struct WorkItem {
    int *pattern;
    int pattern_length;
    int utility;
    int max_item;
    int primary;
    int *work_done;
    int work_count;
};

struct AtomicWorkStack {
    WorkItem items[CAPACITY];
    unsigned int top;
    volatile unsigned int active;  // volatile so threads always reload its value
    
    // Add a lock variable (0 = unlocked, 1 = locked).
    volatile int lock;

    __device__ __host__ void init() {
        top = 0;
        active = 0;
        lock = 0;
    }
    
    __host__ void init_host() {
        top = 0;
        active = 0;
        lock = 0;
    }

    // Simple device-side spinlock using atomicCAS.
    __device__ void acquire_lock() {
        while (atomicCAS((int *)&lock, 0, 1) != 0) {
            // Busy-wait until the lock is acquired.
        }
    }

    __device__ void release_lock() {
        atomicExch((int *)&lock, 0);
    }
    
    // Push a work item using the lock.
    __device__ bool push(WorkItem item) {
        bool success = false;
        acquire_lock();
        if (top < CAPACITY) {
            items[top] = item;
            top++;
            success = true;
            // Update the global push counter.
            atomicAdd(&pushCount, 1);
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
            top--;
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
        atomicAdd(&popCount, 1);
    }
};

__global__ void stressTestKernel(AtomicWorkStack *stack)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    WorkItem localItem;
    
    // Each thread tracks how many tasks it has added.
    int addedTasks = 0;
    
    // While there is still active work, try to process tasks.
    while (stack->get_active() > 0)
    {
        if (stack->pop(&localItem))
        {
            // While processing a popped item, each thread will add new tasks
            // up to its NEW_TASKS_PER_POP quota.
            if (addedTasks < NEW_TASKS_PER_POP) {
                WorkItem newItem;
                newItem.utility = 0;  // Optionally set a value.
                newItem.pattern = nullptr;
                newItem.pattern_length = 0;
                newItem.max_item = 0;
                newItem.primary = 0;
                newItem.work_done = nullptr;
                newItem.work_count = 0;
                
                // Keep trying to push until successful.
                while (!stack->push(newItem)) {
                    // Optionally report if push fails.
                    // printf("Thread %d: push failed (stack full) for new item\n", tid);
                }
                addedTasks++;
            }
            
            // Mark the popped task as finished.
            stack->finish_task();
        }
        // If pop() fails, just loop again.
    }

    // After active work has been processed, if this thread still hasn't
    // reached its NEW_TASKS_PER_POP quota, push the remainder.
    // printf("TID %d has %d tasks left to add\n", tid, NEW_TASKS_PER_POP - addedTasks);
    while (addedTasks < NEW_TASKS_PER_POP) {
        WorkItem newItem;
        newItem.utility = 0;
        newItem.pattern = nullptr;
        newItem.pattern_length = 0;
        newItem.max_item = 0;
        newItem.primary = 0;
        newItem.work_done = nullptr;
        newItem.work_count = 0;
        
        while (!stack->push(newItem)) {
            // printf("Thread %d: push failed (stack full) for new item\n", tid);
        }
        addedTasks++;
    }
}

int main(void)
{
    // Allocate and initialize the stack using unified (managed) memory.
    AtomicWorkStack *hostStack;
    cudaMallocManaged(&hostStack, sizeof(AtomicWorkStack));
    hostStack->init_host();

    // Calculate expected total pushes.
    // Expected = INITIAL_TASKS + NUM_THREADS * NEW_TASKS_PER_POP.
    int totalPushesExpected = INITIAL_TASKS + NUM_THREADS * NEW_TASKS_PER_POP;
    printf("Expected total pushes: %d\n", totalPushesExpected);

    // Prepopulate the stack with INITIAL_TASKS tasks.
    for (int i = 0; i < INITIAL_TASKS; i++) {
        WorkItem item;
        item.utility = 0;
        item.pattern = nullptr;
        item.pattern_length = 0;
        item.max_item = 0;
        item.primary = 0;
        item.work_done = nullptr;
        item.work_count = 0;
        
        // Use the host lock-less operations since we are in a single-threaded host context.
        if (hostStack->top < CAPACITY) {
            hostStack->items[hostStack->top] = item;
            hostStack->top++;
            hostStack->active++;  // Increment active count.
        }
    }
    // Record the pushes done on the host.
    pushCount += INITIAL_TASKS;
    
    // Launch the kernel.
    // (Since hostStack is in managed memory, it is accessible on device.)
    stressTestKernel<<<NUM_THREADS, 1>>>(hostStack);
    cudaDeviceSynchronize();
    
    // After kernel completes, we can inspect the final state.
    printf("Final active count: %u\n", hostStack->active);
    printf("Final top index: %u\n", hostStack->top);
    printf("Push Count: %u\n", pushCount);
    printf("Pop Count: %u\n", popCount);
    
    cudaFree(hostStack);
    return 0;
}
