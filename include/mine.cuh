#pragma once

#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager
#include "mine.cuh"

#define scale 2

__device__ void printBucketUtil(Item *local_util, int max_item)
{
    for (int i = 0; i < max_item; i++)
    {
        if (local_util[i].util > 0)
        {
            printf("%d:%d ", local_util[i].key, local_util[i].util);
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

__device__ void add_bucket_util(Item *local_util, int table_size, int key, int total_util)
{
    // hash the key
    int idx = hashFunction(key, table_size);

    // find the key
    while (true)
    {
        // we are adding in atomic so do compare and swap for the key
        int old = atomicCAS(&local_util[idx].key, 0, key);
        if (old == key)
        {
            // if the key is already present, add the utility
            atomicAdd(&local_util[idx].util, total_util);
            return;
        }
        else if (old == 0)
        {
            // if the key is not present, add the utility since we cas'd it
            atomicAdd(&local_util[idx].util, total_util);
            return;
        }
        // if the key is not present, find the next slot
        idx = (idx + 1) % (table_size);
    }
}

__device__ void add_pattern(WorkItem *wi, int *high_utility_patterns)
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
                     CudaMemoryManager *mm,
                     int min_util)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // Use shared memory only for values that one block will process together.
    __shared__ WorkItem work_item; // the work-item popped from the queue
    __shared__ bool s_popped;      // did we successfully pop a work-item?

    // Shared copies for data that one block uses to process the work-item.
    __shared__ WorkItem new_work_item;
    __shared__ Transaction *temp_transaction;
    __shared__ Item *local_util;
    __shared__ int num_items;
    __shared__ int num_transactions;

    __shared__ int *hashes;
    __shared__ Item *subtree_util;
    __shared__ int max_item;
    __shared__ int primary_count;

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
            __syncthreads(); // all threads must sync before next iteration
            continue;
        }

        // (2) Thread 0 initializes the new work item.
        if (tid == 0)
        {
            // printf("Primary: %d\n", work_item.primary);
            // printf("Pattern Length: %d\n", work_item.pattern_length);
            memset(&new_work_item, 0, sizeof(WorkItem));
            new_work_item.utility = 0;
            // Allocate and copy the pattern
            new_work_item.pattern = reinterpret_cast<int *>(
                mm->malloc((work_item.pattern_length + 1) * sizeof(int)));
            memcpy(new_work_item.pattern,
                   work_item.pattern,
                   work_item.pattern_length * sizeof(int));
            new_work_item.pattern[work_item.pattern_length] = work_item.primary;
            new_work_item.pattern_length = work_item.pattern_length + 1;

            // printPattern(&new_work_item);
            // printf("Old database\n");
            // printf("NumTransactions: %d\n", work_item.db->numTransactions);
            // for (int i = 0; i < work_item.db->numTransactions; i++)
            // {
            //     Transaction &tran = work_item.db->d_transactions[i];
            //     printf("Utility: %d\tLength: %d\tStart: %p\n", tran.utility, tran.length, tran.data);
            // }
            // printDatabase(work_item.db);

            num_items = 0;
            num_transactions = 0;
            local_util = reinterpret_cast<Item *>(
                mm->malloc(work_item.max_item * scale * sizeof(Item)));
            memset(local_util, 0, work_item.max_item * scale * sizeof(Item));
            temp_transaction = reinterpret_cast<Transaction *>(
                mm->malloc(work_item.db->numTransactions * sizeof(Transaction)));
            memset(temp_transaction, 0, work_item.db->numTransactions * sizeof(Transaction));
        }
        __syncthreads();

        // (4) Process each transaction in parallel.
        for (int i = tid; i < work_item.db->numTransactions; i += blockDim.x)
        {
            Transaction &oldTrans = work_item.db->d_transactions[i];
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
                add_bucket_util(local_util,
                                work_item.max_item * scale,
                                newTrans.data[j].key,
                                total_local_util);
            }
        }
        // After processing all transactions in parallel:
        __syncthreads();

        // --- Add this block ---
        // All threads wait, then thread 0 checks if the pattern qualifies:
        if (tid == 0)
        {
            if (new_work_item.utility >= min_util)
            {
                add_pattern(&new_work_item, d_high_utility_patterns);
            }
        }
        __syncthreads();

        // Now handle the case when there are no surviving transactions:
        if (num_transactions == 0)
        {
            if (tid == 0)
            {
                curr_work_queue->finish_task();
            }
            __syncthreads();
            continue;
        }

        // (6) Allocate memory for the new database in new_work_item.
        if (tid == 0)
        {
            new_work_item.db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));
            new_work_item.db->d_data = reinterpret_cast<Item *>(mm->malloc(num_items * sizeof(Item)));
            new_work_item.db->d_transactions = reinterpret_cast<Transaction *>(
                mm->malloc(num_transactions * sizeof(Transaction)));
            new_work_item.db->numItems = 0;
            new_work_item.db->numTransactions = 0;
        }
        __syncthreads();

        // (7) Allocate and initialize structures for merging and primary item calculations.
        if (tid == 0)
        {
            max_item = 0;
            primary_count = 0;
            hashes = reinterpret_cast<int *>(
                mm->malloc(num_transactions * scale * sizeof(int)));
            memset(hashes, -1, num_transactions * scale * sizeof(int));
            subtree_util = reinterpret_cast<Item *>(
                mm->malloc(work_item.max_item * scale * sizeof(Item)));
            memset(subtree_util, 0, work_item.max_item * scale * sizeof(Item));
        }
        __syncthreads();

        // (8) For each temporary transaction, filter and copy it into the new DB.
        for (int i = tid; i < num_transactions; i += blockDim.x)
        {
            Transaction tempTrans = temp_transaction[i];
            int total_subtree_util = tempTrans.utility;
            int count = 0;
            // Count how many items survive filtering.
            for (int j = 0; j < tempTrans.length; j++)
            {
                int idx = find_item(local_util, work_item.max_item * scale, tempTrans.data[j].key);
                if (idx == -1)
                    continue;
                if (local_util[idx].util >= min_util)
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
                int idx = find_item(local_util, work_item.max_item * scale, tempTrans.data[j].key);
                if (idx == -1)
                    continue;
                if (local_util[idx].util >= min_util)
                {
                    newTrans.data[trans_idx++] = tempTrans.data[j];
                    add_bucket_util(subtree_util,
                                    work_item.max_item * scale,
                                    tempTrans.data[j].key,
                                    total_subtree_util - temp_util);
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
                    break;
                }
                // If the transactions have the same key pattern, merge them.
                if (new_work_item.db->sameKey(old, new_trans_idx))
                {
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

        // (10) Update max_item and primary_count in parallel.
        for (int i = tid; i < work_item.max_item * scale; i += blockDim.x)
        {
            if (local_util[i].util >= min_util)
                atomicAdd(&max_item, 1);
            if (subtree_util[i].util >= min_util)
                atomicAdd(&primary_count, 1);
        }
        __syncthreads();

        // (11) Compact the transactions array (this step is done serially by thread 0).
        if (tid == 0)
        {
            // printf("New database\n");
            // printDatabase(new_work_item.db);
            // for (int i = 0; i < new_work_item.db->numTransactions; i++)
            // {
            //     Transaction &tran = new_work_item.db->d_transactions[i];
            //     printf("Utility: %d\tLength: %d\tStart: %p\n", tran.utility, tran.length, tran.data);
            //     // printf("%d|", tran.utility);
            //     // for (int j = 0; j < tran.length; j++)
            //     // {
            //     //     printf("%d:%d ", tran.data[j].key, tran.data[j].util);
            //     // }
            //     // printf("\n");
            // }

            int compact_index = 0;
            for (int i = 0; i < new_work_item.db->numTransactions; i++)
            {
                if (new_work_item.db->d_transactions[i].data != nullptr)
                {
                    new_work_item.db->d_transactions[compact_index++] =
                        new_work_item.db->d_transactions[i];
                }
            }
            new_work_item.db->numTransactions = compact_index;
            new_work_item.max_item = max_item;
            new_work_item.work_count = primary_count;
            // (12) For every surviving primary in subtree_util, push a new work-item.
            for (int i = 0; i < work_item.max_item * scale; i++)
            {
                if (subtree_util[i].util >= min_util)
                {
                    new_work_item.primary = subtree_util[i].key;
                    curr_work_queue->push(new_work_item);
                }
            }
            curr_work_queue->finish_task();
        }
        __syncthreads();

        // (Optional) Free allocated memory here if necessary.
    }
}
