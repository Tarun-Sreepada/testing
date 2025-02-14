#include <cstdio>
#include <cuda_runtime.h>

#define CAPACITY (128*1024)
#define INITIAL_TASKS (1024)
#define NUM_THREADS 1024
#define NEW_TASKS_PER_POP 128

// Global device counter for new work items, starting at 1000.
__device__ int globalTaskCounter = 1000;

// Global managed counters for push operations.
__managed__ unsigned int pushCount = 0;         // Total successful pushes.
__managed__ unsigned int printedPushCount = 0;  // Total pushes printed.

// We'll still use an array to record per-index pushes (for additional debugging).
__managed__ int *appeared;  

// https://secondboyet.com/Articles/LockfreeStack.html

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

    __device__ __host__ void init() {
        top = 0;
        active = 0;
    }
    
    // Push a work item using a CAS loop.
    __device__ bool push(WorkItem item) {
        unsigned int oldTop, newTop;
        do {
            oldTop = top;
            if (oldTop >= CAPACITY) {
                return false;
            }
            newTop = oldTop + 1;
        } while (atomicCAS(&top, oldTop, newTop) != oldTop);
        
        // Write the item into the slot.
        items[oldTop] = item;
        
        // Record that a push occurred at this index.
        atomicAdd(&appeared[oldTop], 1);
        
        // Update the global push counter.
        atomicAdd(&pushCount, 1);
        
        // Print the push event.
        printf("Pushed item at index %d with utility %d\n", oldTop, item.utility);
        // Update the printed push counter.
        atomicAdd(&printedPushCount, 1);
        
        // Increment 'active'
        atomicAdd((unsigned int *)&active, 1);
        // unsigned int oldActive, newActive;
        // do {
        //     oldActive = active;
        //     newActive = oldActive + 1;
        // } while (atomicCAS((unsigned int *)&active, oldActive, newActive) != oldActive);
        
        return true;
    }

    // Pop a work item using a CAS loop.
    __device__ bool pop(WorkItem *item) {
        unsigned int oldTop, newTop;
        do {
            oldTop = top;
            if (oldTop == 0) {
                return false;
            }
            newTop = oldTop - 1;
        } while (atomicCAS(&top, oldTop, newTop) != oldTop);

        printf("Popped item at index %d\n", newTop);
        *item = items[newTop];
        return true;
    }

    __device__ int get_active() {
        __threadfence();
        return atomicAdd((unsigned int *)&active, 0);
    }

    // Mark a task as finished (decrement 'active')
    __device__ void finish_task() {
        atomicSub((unsigned int *)&active, 1);
    }
    
    __host__ void init_host() {
        top = 0;
        active = 0;
    }
};

__global__ void stressTestKernel(AtomicWorkStack *stack)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    WorkItem localItem;

    // Each thread pushes NEW_TASKS_PER_POP new work items into the stack.
    for (int i = 0; i < NEW_TASKS_PER_POP; i++) {
        WorkItem newItem;
        // Use an atomic add on the global counter for a unique utility value.
        newItem.utility = atomicAdd(&globalTaskCounter, 1);
        newItem.pattern = nullptr;
        newItem.pattern_length = 0;
        newItem.max_item = 0;
        newItem.primary = 0;
        newItem.work_done = nullptr;
        newItem.work_count = 0;
        
        // Keep trying to push until successful.
        while (!stack->push(newItem)) {
            // In case the stack is full.
            printf("Thread %d: push failed (stack full) for new item %d\n", tid, newItem.utility);
        }
    }
    
    // Continue processing while there is active work.
    while (stack->get_active() > 0)
    {
        if (!stack->pop(&localItem))
        {
            // If pop failed, try again.
            continue;
        }
        
        // Simulate processing the work item.
        int result = localItem.utility * 2; // Dummy computation
        (void)result; // Suppress unused variable warning
        
        // Mark the popped work item as finished.
        stack->finish_task();
    }
}

int main(void)
{
    // Allocate and initialize the host stack.
    AtomicWorkStack hostStack;
    hostStack.init_host();

    // Calculate total size for pushes.
    int totalPushesExpected = INITIAL_TASKS + NUM_THREADS * NEW_TASKS_PER_POP;
    printf("Expected total pushes: %d\n", totalPushesExpected);

    // Allocate the "appeared" array with CAPACITY elements.
    cudaMallocManaged(&appeared, CAPACITY * sizeof(int));
    cudaMemset(appeared, 0, CAPACITY * sizeof(int));

    // Allocate the device stack.
    AtomicWorkStack *d_stack;
    cudaMalloc((void**)&d_stack, sizeof(AtomicWorkStack));
    cudaMemcpy(d_stack, &hostStack, sizeof(AtomicWorkStack), cudaMemcpyHostToDevice);

    // Populate the stack with INITIAL_TASKS tasks.
    for (int i = 0; i < INITIAL_TASKS; i++) {
        WorkItem item;
        item.utility = 0;
        item.pattern = nullptr;
        item.pattern_length = 0;
        item.max_item = 0;
        item.primary = 0;
        item.work_done = nullptr;
        item.work_count = 0;
        
        if (hostStack.top < CAPACITY) {
            hostStack.items[hostStack.top] = item;
            hostStack.top++;
            hostStack.active++;  // Increment active count.
        }
    }
    
    // Copy the pre-populated stack to the device.
    cudaMemcpy(d_stack, &hostStack, sizeof(AtomicWorkStack), cudaMemcpyHostToDevice);

    // Launch the kernel with NUM_THREADS threads (one thread per block).
    stressTestKernel<<<NUM_THREADS, 1>>>(d_stack);
    cudaDeviceSynchronize();
    
    // Copy back the stack to inspect final state.
    cudaMemcpy(&hostStack, d_stack, sizeof(AtomicWorkStack), cudaMemcpyDeviceToHost);
    printf("Final active count: %u\n", hostStack.active);
    printf("Final top index: %u\n", hostStack.top);

    // Print the counters.
    printf("Global pushCount (every push): %u\n", pushCount);
    printf("Printed push count (push messages printed): %u\n", printedPushCount);

    // Optionally, print the sum of the "appeared" array over CAPACITY.
    unsigned int sumAppeared = 0;
    for (int i = 0; i < CAPACITY; i++) {
        sumAppeared += appeared[i];
    }
    printf("Sum of appeared array: %u\n", sumAppeared);
    
    cudaFree(d_stack);
    cudaFree(appeared);
    return 0;
}
