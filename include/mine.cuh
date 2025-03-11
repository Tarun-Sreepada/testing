#pragma once

#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager
#include "mine.cuh"

#include <cub/cub.cuh>

#define scale 2

#define blocks 1
#define threads 512

#define LOCAL_UTIL true
#define SUBTREE_UTIL false

struct time_stuff
{
    uint64_t start,
        end,
        idle,
        scanning,
        memory_alloc,
        merging,
        push, prev,
        processed_count,
        largest_trans_scan, largest_trans_merge,
        tt_scan, tt_merging;
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
                     int min_util, time_stuff *d_time_stuff)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    // Use shared memory only for values that one block will process together.
    __shared__ WorkItem work_item; // the work-item popped from the queue
    __shared__ Database work_item_db;
    __shared__ bool popped; // did we successfully pop a work-item?

    // Shared copies for data that one block uses to process the work-item.
    __shared__ WorkItem new_work_item;

    __shared__ int num_items;
    __shared__ int num_transactions;

    __shared__ Utils *lu_su;
    __shared__ Transaction *temp_transactions;

    __shared__ int transactions_processed;
    __shared__ Transaction buffer_transactions[threads];
    __shared__ int last_transaction;
    __shared__ int last_item;

    __shared__ int indices[threads + 1]; // for 0
    __shared__ Item buffer_items[threads];
    __shared__ int16_t valid[threads + 1];


    // The outer loop: each iteration pops a new work-item from the global queue.
    while (curr_work_queue->get_active() > 0)
    {
        // (1) Only thread 0 pops from the work queue.
        if (tid == 0)
        {
            popped = curr_work_queue->pop(&work_item);
        }
        __syncthreads();

        // (1.1) If no work-item was popped, then skip this iteration.
        if (!popped)
        {
            __syncthreads(); // all threads must sync before next iteration ? do we need this?
            continue;
        }

        // (2) Thread 0 initializes the new work item.
        if (tid == 0)
        {
            work_item_db = *work_item.db;
            memset(&new_work_item, 0, sizeof(WorkItem));
            printf("Item: %d\n", work_item.primary);

            // Allocate and copy the pattern
            new_work_item.pattern = reinterpret_cast<int *>(global_malloc((work_item.pattern_length + 1) * sizeof(int)));
            memcpy(new_work_item.pattern, work_item.pattern, work_item.pattern_length * sizeof(int));
            new_work_item.pattern[work_item.pattern_length] = work_item.primary;
            new_work_item.pattern_length = work_item.pattern_length + 1;

            lu_su = reinterpret_cast<Utils *>(global_malloc(work_item.max_item * scale * sizeof(Utils)));
            memset(lu_su, 0, work_item.max_item * scale * sizeof(Utils));

            temp_transactions = reinterpret_cast<Transaction *>(
                global_malloc(work_item_db.numTransactions * sizeof(Transaction)));
            memset(temp_transactions, 0, work_item_db.numTransactions * sizeof(Transaction));

            // transactions_processed = 0;

            num_items = 0;
            num_transactions = 0;
            // last_transaction = -1;
        }
        __syncthreads();

        // NOTE: THIS IS WAAAAAY TOO SLOW BUT IM LEAVING IT HERE FOR OTHERS TO SEE IF THEY CAN OPTIMIZE IT
        // // (3) Process each transaction in parallel.
        // while (transactions_processed != work_item.db->numTransactions)
        // {
        //     // Testing how long it would potentially take to scan
        //     // if (tid == 0)
        //     // {
        //     //     // atomicAdd(&transactions_processed, last_transaction + 1);
        //     //     atomicAdd(&transactions_processed, 16384);
        //     // }
        //     // __syncthreads();

        //     // load into buffer
        //     if (tid + transactions_processed < work_item_db.numTransactions)
        //     {
        //         // Copy the transaction to shared memory
        //         buffer_transactions[tid] = work_item_db.d_transactions[tid + transactions_processed];
        //         atomicMax(&last_transaction, tid);
        //     }
        //     __syncthreads();

        //     // load the item indices into buffer
        //     if (tid <= last_transaction)
        //     {
        //         indices[tid + 1] = buffer_transactions[tid].length;
        //     }
        //     __syncthreads();

        //     // if (tid == 0)
        //     // {
        //     //     // atomicAdd(&transactions_processed, last_transaction + 1);
        //     //     atomicAdd(&transactions_processed, 16384);
        //     // }
        //     // __syncthreads();

        //     // cumulative sum
        //     if (tid == 0)
        //     {
        //         // TODO: Use cub for this or implement a parallel prefix sum
        //         indices[0] = 0;
        //         for (int i = 1; i <= last_transaction; i++)
        //         {
        //             indices[i] += indices[i - 1];
        //             // if greater than threads, break
        //             if (indices[i] > threads)
        //             {
        //                 last_transaction = i - 1;
        //                 break;
        //             }
        //         }

        //         atomicAdd(&transactions_processed, last_transaction + 1);
        //         // printf("Transactions Processed: %d\tLast Transaction: %d\tNumber of transactions: %d\n", transactions_processed, last_transaction, work_item.db->numTransactions);
        //     }
        //     __syncthreads();

        //     // get which transaction and which item to load into buffer
        //     int transaction_id = -1;
        //     int item_id = -1;
        //     for (int i = 0; i <= last_transaction; i++)
        //     {
        //         // if (tid >= start_end[i].start && tid < start_end[i].end)
        //         if (tid >= indices[i] && tid < indices[i + 1])
        //         {
        //             transaction_id = i;
        //             item_id = tid - indices[i];
        //             break;
        //         }
        //     }

        //     __syncthreads();

        //     // load the items into buffer
        //     if (transaction_id != -1 && item_id != -1)
        //         buffer_items[tid] = work_item_db.d_transactions[transaction_id].data[item_id];
        //     __syncthreads();

        //     // // if the item is the primary item, process it
        //     if (buffer_items[tid].key == work_item.primary && transaction_id != -1 && item_id != -1)
        //     {
        //         int new_util = buffer_items[tid].value + buffer_transactions[transaction_id].utility;
        //         atomicAdd(&new_work_item.utility, new_util);

        //         // calculate number of items after this item
        //         int num_items_after = buffer_transactions[transaction_id].length - item_id - 1;

        //         // create a new temporary transaction that points to the next item
        //         atomicAdd(&num_items, num_items_after);
        //         int new_tid = atomicAdd(&num_transactions, 1);
        //         temp_transactions[new_tid].length = num_items_after;
        //         temp_transactions[new_tid].utility = buffer_transactions[transaction_id].utility + buffer_items[tid].value;
        //         temp_transactions[new_tid].data = buffer_transactions[transaction_id].data + item_id + 1;

        //         // get TWU for this transaction
        //         for (int i = 0; i < num_items_after; i++)
        //         {
        //             new_util += buffer_items[tid + i + 1].value;
        //         }

        //         // add local utility
        //         for (int i = 0; i < num_items_after; i++)
        //         {
        //             add_bucket_util(lu_su, work_item.max_item * scale, buffer_items[tid + i + 1].key, new_util, LOCAL_UTIL);
        //         }
        //     }
        //     __syncthreads();

        //     last_transaction = -1;

        // }

        for (int i = tid; i < work_item_db.numTransactions; i += threads)
        {
            Transaction t = work_item_db.d_transactions[i];
            int idx = t.findItem(work_item.primary);
            if (idx != -1)
            {
                int new_util = t[idx].value + t.utility;
                atomicAdd(&new_work_item.utility, new_util);

                // calculate number of items after this item
                int num_items_after = t.length - idx - 1;
                if (num_items_after == 0)
                {
                    continue;
                }

                // create a new temporary transaction that points to the next item
                atomicAdd(&num_items, num_items_after);

                int new_tid = atomicAdd(&num_transactions, 1);
                temp_transactions[new_tid].length = num_items_after;
                temp_transactions[new_tid].utility = t.utility + t[idx].value;
                temp_transactions[new_tid].data = t.data + idx + 1;

                // get TWU for this transaction
                for (int j = 0; j < num_items_after; j++)
                {
                    new_util += t[idx + j + 1].value;
                }

                // add local utility
                for (int j = 0; j < num_items_after; j++)
                {
                    add_bucket_util(lu_su, work_item.max_item * scale, t[idx + j + 1].key, new_util, LOCAL_UTIL);
                }
            }
        }

        __syncthreads();

        // (4) Save patterns with utility greater than min_util
        if (tid == 0)
        {
            if (new_work_item.utility > min_util)
            {
                add_pattern(&new_work_item, d_high_utility_patterns);
            }
        }

        // if no new items, skip
        if (num_items == 0)
        {
            if (tid == 0)
            {
                global_free(lu_su);
                global_free(temp_transactions);
                curr_work_queue->finish_task();

                // Free the memory if all work is done using that db
                int ret = atomicAdd(&work_item.work_done[0], 1);
                if (ret == work_item.work_count - 1)
                {
                    global_free(work_item.work_done);
                    global_free(work_item.pattern);
                    global_free(work_item.db->d_data);
                    global_free(work_item.db->d_transactions);
                    global_free(work_item.db);
                }
            }

            __syncthreads();
            continue;
        }

        // (5) Allocate memory for new database
        if (tid == 0)
        {
            printf("Utility: %d\n", new_work_item.utility);

            new_work_item.db = reinterpret_cast<Database *>(global_malloc(sizeof(Database)));
            new_work_item.db->d_data = reinterpret_cast<Item *>(global_malloc(num_items * sizeof(Item)));
            new_work_item.db->d_transactions = reinterpret_cast<Transaction *>(
                global_malloc(num_transactions * sizeof(Transaction)));
            new_work_item.db->numItems = 0;
            new_work_item.db->numTransactions = 0;

            Item *hashes = reinterpret_cast<Item *>(global_malloc(num_transactions * scale * sizeof(Item)));
            memset(hashes, -1, num_transactions * scale * sizeof(Item));

            last_transaction = -1;
            transactions_processed = 0;
        }
        __syncthreads();

        // // (5) Copy the transactions
        // while (transactions_processed != num_transactions)
        // {
        //     __syncthreads();
        //     // load into buffer
        //     if (tid + transactions_processed < num_transactions)
        //     {
        //         buffer_transactions[tid] = temp_transactions[tid + transactions_processed];
        //         atomicMax(&last_transaction, tid);
        //     }
        //     __syncthreads();

        //     // load the item indices into buffer
        //     if (tid <= last_transaction)
        //     {
        //         indices[tid + 1] = buffer_transactions[tid].length;
        //     }
        //     __syncthreads();

        //     // cumulative sum
        //     if (tid == 0)
        //     {
        //         indices[0] = 0;
        //         // printf("%d ", indices[0]);
        //         for (int i = 1; i <= last_transaction; i++)
        //         {
        //             // printf("%d ", indices[i]);
        //             indices[i] += indices[i - 1];
        //             // if greater than threads, break
        //             if (indices[i] > threads)
        //             {
        //                 last_transaction = i - 1;
        //                 break;
        //             }
        //         }
        //         // printf("\n");

        //         // // print the cumulative sum
        //         // for (int i = 0; i <= last_transaction; i++)
        //         // {
        //         //     printf("%d:%d ", i,indices[i]);
        //         // }
        //         // printf("\n");

        //         transactions_processed += last_transaction + 1;
        //     }
        //     __syncthreads();

        //     // get which transaction and which item to load into buffer
        //     int transaction_id = -1;
        //     int item_id = -1;
        //     for (int i = 0; i < last_transaction + 1; i++)
        //     {
        //         if (tid >= indices[i] && tid < indices[i + 1] && indices[i + 1] < threads)
        //         {
        //             transaction_id = i;
        //             item_id = tid - indices[i];
        //             break;
        //         }
        //     }


        //     __syncthreads();
        //     // printf("TID:%d\tTransaction ID: %d, Item ID: %d\n", tid,transaction_id, item_id);
        //     __syncthreads();

        //     // __syncthreads();
        //     valid[tid + 1] = false;
        //     // if(tid == 0)
        //     // {
        //     //     printf("Last Transaction: %d\n\n", last_transaction);
        //     // }
        //     // __syncthreads();

        //     // load the items into buffer
        //     if (transaction_id != -1 && item_id != -1)
        //     {
        //         buffer_items[tid] = temp_transactions[transaction_id].data[item_id];
        //         atomicMax(&last_item, tid);
        //         // if the item had a local utility greater than min_util, set valid to true
        //         int idx = find_item(lu_su, work_item.max_item * scale, buffer_items[tid].key);
        //         if (idx != -1 && lu_su[idx].local_util >= min_util)
        //         {
        //             valid[tid] = true;
        //         }
        //     }
        //     __syncthreads();

        //     // if (tid == 0)
        //     // {
        //     //     printf("Last Transaction: %d\n", last_transaction);
        //     //     printf("Last Item: %d\n", last_item);
        //     //     for (int i = 0; i < last_item; i++)
        //     //     {
        //     //         printf("%d:%d ", buffer_items[i].key, buffer_items[i].value);
        //     //     }
        //     //     printf("\n");

        //     //     for (int i = 0; i < last_item; i++)
        //     //     {
        //     //         printf("%d ", valid[i]);
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // __syncthreads();

        //     // cumulative sum of the valid items
        //     if (tid == 0)
        //     {
        //         valid[0] = 0;
        //         for (int i = 1; i <= last_item; i++)
        //         {
        //             valid[i] += valid[i - 1];
        //         }
        //     }

            

        //     // // print the cumulative sum
        //     // if (tid == 0)
        //     // {
        //     //     for (int i = 0; i < last_item; i++)
        //     //     {
        //     //         printf("%d ", valid[i]);
        //     //     }
        //     //     printf("\n");
        //     // }

        //     if (tid == 0)
        //     {
        //         last_transaction = -1;
        //         last_item = -1;
        //         // printf("\n\n");
        //     }

        //     __syncthreads();


        // }
        // __syncthreads();

        // (6) Merge identical transactions

        // (7) Create new work items
        if (tid == 0)
        {
            global_free(lu_su);
            global_free(temp_transactions);
            curr_work_queue->finish_task();
        }
        __syncthreads();
    }
}
