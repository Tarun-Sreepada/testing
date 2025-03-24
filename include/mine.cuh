#pragma once

#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager
#include "mine.cuh"

#include <cub/cub.cuh>

#define scale 2

// #define blocks 128
// #define threads 512

#define LOCAL_UTIL true
#define SUBTREE_UTIL false

#define bucket_per_core 8192

struct time_stuff
{
    uint64_t
        start,                                               // start time
        end,                                                 // end time
        idle,                                                // idle time
        scanning,                                            // scanning time
        memory_alloc,                                        // memory allocation time
        merging,                                             // write and merging time
        push,                                                // push time
        prev,                                                // previous time
        processed_count,                                     // number of work items processed
        largest_trans_scan, largest_trans_merge,             // largest transaction length
        largest_trans_count_scan, largest_trans_count_merge, // number of transactions
        merge_count,                                         // number of merged transactions
        tt_scan, tt_merging;                                 // total time for scanning and merging
};

struct flame_graph_bucket {
    uint64_t idle;           // Sum of idle time for these iterations
    uint64_t scanning;       // Sum of scanning time for these iterations
    uint64_t memory_alloc;   // Sum of memory allocation time for these iterations
    uint64_t merging;        // Sum of merging time for these iterations
    uint64_t push;           // Sum of push time for these iterations
};

struct core_flame_graph {
    struct flame_graph_bucket buckets[bucket_per_core];
};



__device__ void printBucketUtil(Utils *lu_su, int max_item)
{
    for (int i = 0; i < max_item; i++)
    {
        if (lu_su[i].local_util > 0 || lu_su[i].subtree_util > 0)
        {
            printf("Key: %d, Local Util: %d, Subtree Util: %d\n", lu_su[i].key, lu_su[i].local_util, lu_su[i].subtree_util);
        }
    }
    printf("\n");
}

__device__ void printPattern(WorkItem *wi)
{
    printf("Pattern: ");
    for (int i = 0; i < wi->pattern_length; i++)
    {
        printf("%d ", wi->pattern[i]);
    }
    printf("\n");
}

__device__ void add_bucket_util(Utils *lu_su, int table_size, int key, int total_util, bool lu_or_su)
{
    // hash the key
    int idx = hashFunction(key, table_size);

    // find the key
    while (true)
    {
        // we are adding in atomic so do compare and swap for the key
        int old = atomicCAS(&lu_su[idx].key, 0, key);
        if (old == key || old == 0)
        {
            if (lu_or_su == LOCAL_UTIL)
            {
                atomicAdd(&lu_su[idx].local_util, total_util);
            }
            else
            {
                atomicAdd(&lu_su[idx].subtree_util, total_util);
            }
            return;
        }
        // if the key is not present, find the next slot
        idx = (idx + 1) % (table_size);
    }
}

__device__ __noinline__ void add_pattern(WorkItem *wi, int *high_utility_patterns)
{
    // printf("Pattern Count:%d\n", atomicAdd(&high_utility_patterns[0], 1));

    int idx = atomicAdd(&high_utility_patterns[1], (wi->pattern_length + 2));

    // printPattern(wi);
    // printf("Utility: %d\n", wi->utility);

    for (int i = 0; i < wi->pattern_length; i++)
    {
        high_utility_patterns[idx + i] = wi->pattern[i];
    }
    high_utility_patterns[idx + wi->pattern_length] = wi->utility;
}

__global__ void test(AtomicWorkStack *curr_work_queue,
                     int32_t *d_high_utility_patterns,
                     int min_util, time_stuff *d_time_stuff,
                     core_flame_graph *flame_graph)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // Use shared memory only for values that one block will process together.
    __shared__ WorkItem work_item; // the work-item popped from the queue
    __shared__ bool s_popped;      // did we successfully pop a work-item?

    // Shared copies for data that one block uses to process the work-item.
    __shared__ WorkItem new_work_item;
    __shared__ Transaction *temp_transaction;

    __shared__ int num_items;
    __shared__ int num_transactions;

    __shared__ int *hashes;
    __shared__ int max_item;
    __shared__ int primary_count;

    __shared__ Utils *lu_su;

    __shared__ int largest_trans_scan;
    __shared__ int largest_trans_merge;

    __shared__ int merge_count;

    __shared__ time_stuff local_ts;

    extern __shared__ char shared_memory[];

    lu_su = reinterpret_cast<Utils *>(shared_memory);


    if (tid == 0)
    {
        memset(&local_ts, 0, sizeof(time_stuff));
        local_ts.start = clock64();
        local_ts.prev = local_ts.start;
    }
    __syncthreads();


    // The outer loop: each iteration pops a new work-item from the global queue.
    while (curr_work_queue->get_active() > 0)
    {
        // (1) Only thread 0 pops from the work queue.
        if (tid == 0)
        {
            s_popped = curr_work_queue->pop(&work_item);
        }
        __syncthreads();

        // If no work-item was popped, then skip this iteration.
        if (!s_popped)
        {
            if (tid == 0)
            {
                uint64_t curr_time = clock64();
                uint64_t diff = curr_time - local_ts.prev;

                local_ts.idle += diff;
                local_ts.prev = curr_time;

                // local_ts.idle += curr_time - local_ts.prev;
                // local_ts.prev = curr_time;
                flame_graph[bid].buckets[local_ts.processed_count].idle += diff;
            }
            __syncthreads(); // all threads must sync before next iteration
            continue;
        }

        // (2) Thread 0 initializes the new work item.
        if (tid == 0)
        {
            memset(&new_work_item, 0, sizeof(WorkItem));

            // Allocate and copy the pattern
            new_work_item.pattern = reinterpret_cast<int *>(global_malloc((work_item.pattern_length + 1) * sizeof(int)));
            // memcpy(new_work_item.pattern, work_item.pattern, work_item.pattern_length * sizeof(int));
            new_work_item.pattern[work_item.pattern_length] = work_item.primary;
            new_work_item.pattern_length = work_item.pattern_length + 1;

            num_items = 0;
            num_transactions = 0;

            // memset(lu_su, 0, work_item.max_item * scale * sizeof(Utils));

            temp_transaction = reinterpret_cast<Transaction *>(
                global_malloc(work_item.db->numTransactions * sizeof(Transaction)));
            // memset(temp_transaction, 0, work_item.db->numTransactions * sizeof(Transaction));

            largest_trans_scan = 0;
            largest_trans_merge = 0;
        }
        __syncthreads();

        for (int i = tid; i < work_item.pattern_length; i += blockDim.x)
        {
            new_work_item.pattern[i] = work_item.pattern[i];
        }

        __syncthreads();

        for (int i = tid; i < work_item.max_item * scale; i += blockDim.x)
        {
            lu_su[i].key = 0;
            lu_su[i].local_util = 0;
            lu_su[i].subtree_util = 0;
        }

        __syncthreads();

        for (int i = tid; i < work_item.db->numTransactions; i += blockDim.x)
        {
            memset(&temp_transaction[i], 0, sizeof(Transaction));
        }
        __syncthreads();

        if (tid == 0)
        {
            // local_ts.memory_alloc += clock64() - local_ts.prev;
            // local_ts.prev = clock64();
            uint64_t curr_time = clock64();
            uint64_t diff = curr_time - local_ts.prev;

            local_ts.memory_alloc += diff;
            local_ts.prev = curr_time;

            flame_graph[bid].buckets[local_ts.processed_count].memory_alloc += diff;
        }
        __syncthreads();

        // (4) Process each transaction in parallel.
        for (int i = tid; i < work_item.db->numTransactions; i += blockDim.x)
        {
            Transaction &oldTrans = work_item.db->d_transactions[i];
            // atomicMax(&local_ts.largest_trans, (int)oldTrans.length);
            atomicMax((int *)&largest_trans_scan, (int)oldTrans.length);
            int idx = oldTrans.findItem(work_item.primary);
            if (idx == -1)
                continue;

            // Update new_work_item.utility atomically.
            atomicAdd(&new_work_item.utility, oldTrans.utility + oldTrans.data[idx].util);

            int suffix_count = oldTrans.length - (idx + 1);
            if (suffix_count <= 0)
                continue;

            // Reserve space for suffix items.
            atomicAdd(&num_items, suffix_count);
            int t_idx = atomicAdd(&num_transactions, 1);

            // Create a temporary transaction with the suffix.
            Transaction &newTrans = temp_transaction[t_idx];
            newTrans.data = oldTrans.data + idx + 1;
            newTrans.length = suffix_count;
            newTrans.utility = oldTrans.utility + oldTrans.data[idx].util;

            // Compute the total local utility for the transaction.
            int total_local_util = newTrans.utility;
            for (int j = 0; j < suffix_count; j++)
            {
                total_local_util += newTrans.data[j].util;
            }

            // Update the local utility buckets.
            for (int j = 0; j < suffix_count; j++)
            {
                // add_bucket_util(local_util,
                //                 work_item.max_item * scale,
                //                 newTrans.data[j].key,
                //                 total_local_util);
                add_bucket_util(lu_su, work_item.max_item * scale, newTrans.data[j].key, total_local_util, LOCAL_UTIL);
            }
        }
        // After processing all transactions in parallel:
        __syncthreads();

        // --- Add this block ---
        // All threads wait, then thread 0 checks if the pattern qualifies:
        if (tid == 0)
        {
            uint64_t curr_time = clock64();
            uint64_t diff = curr_time - local_ts.prev;

            local_ts.scanning += diff;

            flame_graph[bid].buckets[local_ts.processed_count].scanning += diff;

            if (local_ts.tt_scan < diff)
            {
                local_ts.tt_scan = diff;
                local_ts.largest_trans_scan = largest_trans_scan;
                local_ts.largest_trans_count_scan = work_item.db->numTransactions;
            }

            local_ts.prev = curr_time;
            // local_ts.scanning += clock64() - local_ts.prev;
            // local_ts.prev = clock64();
            // printBucketUtil(lu_su, work_item.max_item * scale);
            // printf("Num new items: %d\n", num_items);
            // printf("Num new transactions: %d\n", num_transactions);
            if (new_work_item.utility >= min_util)
            {
                add_pattern(&new_work_item, d_high_utility_patterns);
            }
        }
        __syncthreads();

        // // Now handle the case when there are no surviving transactions:
        if (num_transactions == 0)
        {
            if (tid == 0)
            {
                curr_work_queue->finish_task();

                // Free allocated memory here if necessary.
                // global_free(local_util);
                global_free(temp_transaction);

                // Free the pattern.
                global_free(new_work_item.pattern);

                int ret = atomicAdd(&work_item.work_done[0], 1);
                if (ret == work_item.work_count - 1)
                {
                    global_free(work_item.work_done);
                    global_free(work_item.pattern);
                    global_free(work_item.db->d_data);
                    global_free(work_item.db->d_transactions);
                    global_free(work_item.db);
                }

                local_ts.processed_count++;

            }
            __syncthreads();
            continue;
        }

        // // (6) Allocate memory for the new database in new_work_item.
        if (tid == 0)
        {
            // printf("Allocating memory\n");
            new_work_item.db = reinterpret_cast<Database *>(global_malloc(sizeof(Database)));
            new_work_item.db->d_data = reinterpret_cast<Item *>(global_malloc(num_items * sizeof(Item)));
            new_work_item.db->d_transactions = reinterpret_cast<Transaction *>(
                global_malloc(num_transactions * sizeof(Transaction)));
            new_work_item.db->numItems = 0;
            new_work_item.db->numTransactions = 0;

            max_item = 0;
            primary_count = 0;
            hashes = reinterpret_cast<int *>(
                global_malloc(num_transactions * scale * sizeof(int)));
            // memset(hashes, -1, num_transactions * scale * sizeof(int));

            merge_count = 0;

            uint64_t curr_time = clock64();
            uint64_t diff = curr_time - local_ts.prev;

            // local_ts.memory_alloc += clock64() - local_ts.prev;
            // local_ts.prev = clock64();

            local_ts.memory_alloc += diff;
            local_ts.prev = curr_time;

            flame_graph[bid].buckets[local_ts.processed_count].memory_alloc += diff;
        }
        __syncthreads();

        for (int i = tid; i < num_transactions * scale; i += blockDim.x)
        {
            hashes[i] = -1;
        }

        __syncthreads();

        for (int i = tid; i < num_transactions; i += blockDim.x)
        {

            Transaction tempTrans = temp_transaction[i];
            atomicMax((int *)&largest_trans_merge, (int)tempTrans.length);
            int total_subtree_util = tempTrans.utility;
            int count = 0;
            // Count how many items survive filtering.
            for (int j = 0; j < tempTrans.length; j++)
            {
                // int idx = find_item(local_util, work_item.max_item * scale, tempTrans.data[j].key);
                int idx = find_item(lu_su, work_item.max_item * scale, tempTrans.data[j].key);
                // if (local_util[idx].util >= min_util)
                if (lu_su[idx].local_util >= min_util)
                {
                    count++;
                    total_subtree_util += tempTrans.data[j].util;
                }
            }
            if (count == 0)
                continue;

            int new_trans_idx = atomicAdd(&new_work_item.db->numTransactions, 1);
            Transaction &newTrans = new_work_item.db->d_transactions[new_trans_idx];

            int start = atomicAdd(&new_work_item.db->numItems, count);
            newTrans.data = new_work_item.db->d_data + start;
            newTrans.length = count;
            newTrans.utility = tempTrans.utility;

            int trans_idx = 0;
            int temp_util = 0;
            for (int j = 0; j < tempTrans.length; j++)
            {
                // int idx = find_item(local_util, work_item.max_item * scale, tempTrans.data[j].key);
                int idx = find_item(lu_su, work_item.max_item * scale, tempTrans.data[j].key);
                // if (local_util[idx].util >= min_util)
                if (lu_su[idx].local_util >= min_util)
                {
                    newTrans.data[trans_idx++] = tempTrans.data[j];
                    add_bucket_util(lu_su, work_item.max_item * scale, tempTrans.data[j].key, total_subtree_util - temp_util, SUBTREE_UTIL);
                    temp_util += tempTrans.data[j].util;
                }
            }

            // (9) Use open addressing to try to merge transactions.
            int hash_idx = items_hasher(newTrans.data, newTrans.length, num_transactions * scale);
            while (true)
            {
                int old = atomicCAS(&hashes[hash_idx], -1, new_trans_idx);
                if (old == -1)
                {
                    // Inserted successfully.
                    // cumulative_indices[new_trans_idx + 1] = 1;
                    break;
                }
                // If the transactions have the same key pattern, merge them.
                if (new_work_item.db->sameKey(old, new_trans_idx))
                {
                    atomicAdd(&merge_count, 1);
                    // printf("Merge count: %d\n", atomicAdd(&merge_count, 1));
                    Transaction &oldTran = new_work_item.db->d_transactions[old];
                    atomicAdd(&oldTran.utility, newTrans.utility);
                    for (int j = 0; j < oldTran.length; j++)
                    {
                        atomicAdd(&oldTran.data[j].util, newTrans.data[j].util);
                    }
                    // Mark this transaction as merged.
                    newTrans.data = nullptr;
                    newTrans.length = 0;
                    newTrans.utility = 0;
                    break;
                }
                hash_idx = (hash_idx + 1) % (num_transactions * scale);
            }
        }
        __syncthreads();

        // if (new_work_item.db->numTransactions == 0)
        // {
        //     if (tid == 0)
        //     {
        //         // local_ts.merging += clock64() - local_ts.prev;
        //         // local_ts.prev = clock64();

        //         uint64_t curr_time = clock64();
        //         uint64_t diff = curr_time - local_ts.prev;

        //         local_ts.merging += diff;
        //         if (local_ts.tt_merging < diff)
        //         {
        //             local_ts.tt_merging = diff;
        //             local_ts.largest_trans_merge = largest_trans_merge;
        //             local_ts.largest_trans_count_merge = num_transactions;
        //             local_ts.merge_count = merge_count;
        //         }

        //         curr_work_queue->finish_task();

        //         // Free allocated memory here if necessary.
        //         global_free(temp_transaction);
        //         global_free(hashes);
        //         global_free(new_work_item.db->d_data);
        //         global_free(new_work_item.db->d_transactions);
        //         global_free(new_work_item.db);

        //         // Free the pattern.
        //         global_free(new_work_item.pattern);

        //        local_ts.processed_count++;

        //     }
        //     __syncthreads();
        //     continue;
        // }

        // __syncthreads();


        // (10) Update max_item and primary_count in parallel.
        for (int i = tid; i < work_item.max_item * scale; i += blockDim.x)
        {
            if (lu_su[i].local_util >= min_util)
                atomicAdd(&max_item, 1);
            if (lu_su[i].subtree_util >= min_util)
                atomicAdd(&primary_count, 1);
        }
        __syncthreads();

        if (tid == 0)
        {
            num_transactions = 0;
        }

        __syncthreads();

        // push down empty transactions
        if (tid < 32)
        {
            for (int i = tid; i < new_work_item.db->numTransactions; i += 32)
            {
                Transaction copy = new_work_item.db->d_transactions[i];
                if (copy.data != nullptr)
                {
                    int ret = atomicAdd(&num_transactions, 1);
                    new_work_item.db->d_transactions[ret] = copy;
                }
            }
        }

        __syncthreads();

        // sort transactions based on length : TODO


        if (tid == 0)
        {
            new_work_item.db->numTransactions = num_transactions;

            uint64_t curr_time = clock64();
            uint64_t diff = curr_time - local_ts.prev;
            local_ts.merging += diff;
            local_ts.prev = curr_time;

            flame_graph[bid].buckets[local_ts.processed_count].merging += diff;

            if (local_ts.tt_merging < diff)
            {
                local_ts.tt_merging = diff;
                local_ts.largest_trans_merge = largest_trans_merge;
                local_ts.largest_trans_count_merge = num_transactions;
            }

            new_work_item.max_item = max_item;
            new_work_item.work_count = primary_count;
            new_work_item.work_done = reinterpret_cast<int *>(
                global_malloc(sizeof(int)));
            new_work_item.work_done[0] = 0;
        }

        __syncthreads();

        // (12) For every surviving primary in subtree_util, push a new work-item.
        for (int i = tid; i < work_item.max_item * scale; i += blockDim.x)
        {
            if (lu_su[i].subtree_util >= min_util)
            {
                WorkItem copy = new_work_item;
                copy.primary = lu_su[i].key;
                curr_work_queue->push(copy);
            }
        }
        __syncthreads();

        if (tid == 0)
        {
            curr_work_queue->finish_task();

            // Free allocated memory here if necessary.
            global_free(temp_transaction);
            global_free(hashes);

            int ret = atomicAdd(&work_item.work_done[0], 1);

            if (ret == (work_item.work_count - 1))
            {

                global_free(work_item.work_done);
                global_free(work_item.pattern);
                global_free(work_item.db->d_data);
                global_free(work_item.db->d_transactions);
                global_free(work_item.db);
            }

            // local_ts.push += clock64() - local_ts.prev;
            // local_ts.prev = clock64();

            uint64_t curr_time = clock64();
            uint64_t diff = curr_time - local_ts.prev;

            local_ts.push += diff;
            local_ts.prev = curr_time;

            flame_graph[bid].buckets[local_ts.processed_count].push += diff;

            local_ts.processed_count++;

        }

        __syncthreads();

        // (Optional) Free allocated memory here if necessary.
    }

    if (tid == 0)
    {
        uint64_t curr_time = clock64();
        uint64_t diff = curr_time - local_ts.prev;

        local_ts.idle += diff;
        local_ts.prev = curr_time;
        d_time_stuff[bid] = local_ts;

        flame_graph[bid].buckets[local_ts.processed_count].idle += diff;
    }
}