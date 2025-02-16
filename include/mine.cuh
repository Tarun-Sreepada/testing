#pragma once

#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager
#include "mine.cuh"

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "Utility.h"

#define scale 2

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

    for (int i = 0; i < wi->pattern_length; i++)
    {
        high_utility_patterns[idx + i] = wi->pattern[i];
    }
    high_utility_patterns[idx + wi->pattern_length] = wi->utility;
}

__global__ void test(AtomicWorkStack *curr_work_queue, int32_t *d_high_utility_patterns, CudaMemoryManager *mm, int min_util)
{

    __shared__ WorkItem old_work_item;
    __shared__ WorkItem new_work_item;
    __shared__ Item *local_util;
    __shared__ Item *subtree_util;

    __shared__ int item;

    __shared__ int *hashes;
    // __shared__ bool started;
    // started = true;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // printf("Block: %d\tThread: %d\n", bid, tid);

    __shared__ bool popped;

    __shared__ int max_item;
    // int max_item = 0;

    __shared__ int primary_count;
    // int primary_count = 0;

    while (curr_work_queue->get_active() > 0)
    {
        if (tid == 0)
        {
            // curr_work_queue->pop(item);
            popped = curr_work_queue->pop(&old_work_item);
            // if (popped) started = false;
            // printf("BID: %d TID %d is popping\tActive:%d\n", bid, tid, curr_work_queue->get_active());
        }

        __syncthreads();

        if (!popped)
        {
            continue; // skip
        }
        // printf("TID: %d\tPrimary: %d\n", tid, old_work_item.primary);

        if (tid == 0)
        {
            // printf("Primary: %d\n", old_work_item.primary);
            item = old_work_item.primary;

            memset(&new_work_item, 0, sizeof(WorkItem));

            new_work_item.pattern = reinterpret_cast<int *>(mm->malloc((old_work_item.pattern_length + 1) * sizeof(int)));
            memcpy(new_work_item.pattern, old_work_item.pattern, (old_work_item.pattern_length) * sizeof(int));
            new_work_item.pattern[old_work_item.pattern_length] = old_work_item.primary;
            new_work_item.pattern_length = old_work_item.pattern_length + 1;

            new_work_item.db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));
            memset(new_work_item.db, 0, sizeof(Database));
            new_work_item.db->d_data = reinterpret_cast<Item *>(mm->malloc(old_work_item.db->numItems * sizeof(Item)));
            new_work_item.db->d_transactions = reinterpret_cast<Transaction *>(mm->malloc(old_work_item.db->numTransactions * sizeof(Transaction)));

            memset(new_work_item.db->d_data, 0, old_work_item.db->numItems * sizeof(Item));
            memset(new_work_item.db->d_transactions, 0, old_work_item.db->numTransactions * sizeof(Transaction));

            new_work_item.max_item = old_work_item.max_item;
            local_util = reinterpret_cast<Item *>(mm->malloc(new_work_item.max_item * scale * sizeof(Item)));
            memset(local_util, 0, new_work_item.max_item * scale * sizeof(Item));
            subtree_util = reinterpret_cast<Item *>(mm->malloc(new_work_item.max_item * scale * sizeof(Item)));
            memset(subtree_util, 0, new_work_item.max_item * scale * sizeof(Item));

            // started = true;
        }

        __syncthreads();


        // Iterate over all transactions in the old database.
        for (int i = tid; i < old_work_item.db->numTransactions; i += blockDim.x)
        {
            // For easier reference.
            Transaction &oldTrans = old_work_item.db->d_transactions[i];

            int idx = oldTrans.findItem(item);
            if (idx == -1)
            {
                continue;
            }

            // Compute the number of items after the found index.
            int items_this_trans = oldTrans.length - (idx + 1);
            // new_work_item.utility += oldTrans.utility + oldTrans.data[idx].util;
            atomicAdd(&new_work_item.utility, oldTrans.utility + oldTrans.data[idx].util);

            if (items_this_trans <= 0)
            {
                continue;
            }

            // Reserve space in the current (flat) data array.
            // Record the starting index for the new transaction’s data.
            // int new_start = curr->db->numItems;
            // curr->db->numItems += items_this_trans;
            int new_start = atomicAdd(&new_work_item.db->numItems, items_this_trans);

            // Create a new transaction in curr->db.
            int tran_ret = atomicAdd(&new_work_item.db->numTransactions, 1);
            // int tran_ret = curr->db->numTransactions;
            // curr->db->numTransactions += 1;
            Transaction &newTrans = new_work_item.db->d_transactions[tran_ret];

            // Set the new transaction’s utility.
            newTrans.utility = oldTrans.utility + oldTrans.data[idx].util;
            // Set the pointer to the beginning of its block in the flat array.
            newTrans.data = new_work_item.db->d_data + new_start;

            // Compute total utility by summing the utilities of all suffix items.
            int total_util = newTrans.utility;
            for (int j = idx + 1; j < oldTrans.length; j++)
            {
                total_util += oldTrans.data[j].util;
            }

            // Copy the suffix items into the new region.
            int ret = new_start;
            for (int j = idx + 1; j < oldTrans.length; j++)
            {
                new_work_item.db->d_data[ret++] = oldTrans.data[j];
                // Update bucket utility.
                add_bucket_util(local_util, old_work_item.max_item * scale, oldTrans.data[j].key, total_util);
            }
            // Record the new transaction's length.
            newTrans.length = ret - new_start;
        }

        __syncthreads();

        // if utility is greater than min_util add to high utility patterns
        if (tid == 0)
        {
            if (new_work_item.utility >= min_util)
            {
                add_pattern(&new_work_item, d_high_utility_patterns);
            }

            hashes = (int *)(mm->malloc(new_work_item.db->numTransactions * scale * sizeof(int)));
            memset(hashes, -1, new_work_item.db->numTransactions * sizeof(int) * scale);
            // new_work_item->work_done = reinterpret_cast<int *>(mm->malloc(sizeof(int)));
            // printDatabase(new_work_item.db);
            max_item = 0;
            primary_count = 0;
        }
        __syncthreads();

        // // // Loop over each transaction in the current database.
        for (int i = tid; i < new_work_item.db->numTransactions; i += blockDim.x)
        {
            // Reference to the current transaction.
            Transaction &tran = new_work_item.db->d_transactions[i];

        //     // curr_loc tracks the new (trimmed) position within this transaction.
            int curr_loc = 0;
            // Start with the transaction's current utility.
            int total_tran_util = tran.utility;

            // --- Trim Phase ---
            // Iterate over each item in the transaction.
            for (int j = 0; j < tran.length; j++)
            {
                // Find the index into local_util for the key of the current item.
                int idx = find_item(local_util, new_work_item.max_item * scale, tran.data[j].key);
                // If the local utility meets the minimum threshold...
                if (local_util[idx].util >= min_util)
                {
                    // Overwrite the current transaction's data in-place.
                    // (Assumes tran.data points to a modifiable array.)
                    tran.data[curr_loc] = tran.data[j];
                    curr_loc++;
                    total_tran_util += tran.data[j].util; // for subtree
                }
            }
            // Update the transaction length to reflect the trimmed items.
            tran.length = curr_loc;

            // --- Update Subtree Utility ---
            int temp_util = 0;
            for (int j = 0; j < tran.length; j++)
            {
                // add_bucket_util updates the subtree utility for the given key.
                add_bucket_util(subtree_util, new_work_item.max_item * scale, tran.data[j].key, total_tran_util - temp_util);
                temp_util += tran.data[j].util;
            }

            // --- Hashing and Merging ---
            // Compute a hash for the trimmed transaction using its data.
            int hash_idx = items_hasher(tran.data, tran.length, new_work_item.db->numTransactions * scale);

            while (true)
            {
                // Try to insert the current transaction index into the hash table.
                int old = atomicCAS(&hashes[hash_idx], -1, i);
                if (old == -1)
                {
                    // Insertion succeeded.
                    break;
                }
                // If the slot is occupied and the transactions have the same keys...
                if (new_work_item.db->sameKey(old, i))
                {
                    // If merging with itself, do nothing.
                    if (i == old)
                        break;

                    // If the transactions have different lengths, try the next hash slot.
                    if (new_work_item.db->d_transactions[old].length != tran.length)
                    {
                        hash_idx = (hash_idx + 1) % (new_work_item.db->numTransactions * scale);
                        continue;
                    }
                    // --- Merge Phase ---
                    // Merge the two transactions item-by-item.
                    Transaction &oldTran = new_work_item.db->d_transactions[old];
                    for (int j = 0; j < oldTran.length; j++)
                    {
                        // atomicAdd(&oldTran.data[i].util, tran.data[i].util);
                        // oldTran.data[j].util += tran.data[j].util;
                        atomicAdd(&oldTran.data[j].util, tran.data[j].util);
                    }
                    // Merge overall utilities.
                    atomicAdd(&oldTran.utility, tran.utility);
                    // Mark the current transaction as merged by zeroing its fields.
                    tran.data = nullptr;
                    tran.length = 0;
                    tran.utility = 0;
                    break;
                }
                // Otherwise, try the next slot in the hash table.
                hash_idx = (hash_idx + 1) % (new_work_item.db->numTransactions * scale);
            }

        }


        __syncthreads();

        for (int i = tid; i < old_work_item.max_item * scale; i += blockDim.x)
        {
            if (local_util[i].util >= min_util)
                atomicAdd(&max_item, 1);
            if (subtree_util[i].util >= min_util)
                atomicAdd(&primary_count, 1);
        }

        __syncthreads();
        if (primary_count == 0)
        {
            if (tid == 0)
            {
                curr_work_queue->finish_task();
            }
            continue;
        }

        if (tid == 0)
        {
   
            new_work_item.max_item = max_item;
            new_work_item.work_count = primary_count;

            for (int i = 0; i < old_work_item.max_item * scale; i++)
            {
                if (subtree_util[i].util >= min_util)
                {
                    new_work_item.primary = subtree_util[i].key;
                    curr_work_queue->push(new_work_item);
                }
            }

            curr_work_queue->finish_task();
        }
    }
}