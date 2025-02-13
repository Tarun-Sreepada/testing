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

__global__ void project(WorkItem *old, WorkItem *curr, Item *local_util)
{

    int tid = blockIdx.x;

    // if (tid >= old->db->numTransactions)
    //     return;

    // printf("TID: %d\n", tid);

    // find the item in old
    int item = curr->pattern[curr->pattern[0]];
    int idx = old->db->d_transactions[tid].findItem(item);
    if (idx == -1)
    {
        // printf("Item not found\n");
        atomicAdd(&curr->db->transaction_tracker, 1);
        __threadfence_system();

        return;
    }

    int items_this_trans = old->db->d_transactions[tid].end - (idx + 1);
    int ret = atomicAdd(&curr->db->numItems, items_this_trans);
    int tran_ret = atomicAdd(&curr->db->numTransactions, 1);

    curr->db->d_transactions[tran_ret].utility = old->db->d_transactions[tid].utility + old->db->d_data[idx].util;
    curr->db->d_transactions[tran_ret].data = curr->db->d_data;

    // write.

    // update the db
    curr->db->d_transactions[tran_ret].start = ret;

    int total_util = curr->db->d_transactions[tran_ret].utility;
    atomicAdd(&curr->utility, total_util);
    for (int i = idx + 1; i < old->db->d_transactions[tid].end; i++)
    {
        total_util += old->db->d_data[i].util;
    }

    for (int i = idx + 1; i < old->db->d_transactions[tid].end; i++)
    {
        //     ret++;
        curr->db->d_data[ret++] = old->db->d_data[i];
        // printf("TID: %d\tItem: %d\tUtility: %d\n", tid, old->db->d_data[i].key, old->db->d_data[i].util);
        add_bucket_util(local_util, old->max_item * scale, old->db->d_data[i].key, total_util);
    }
    curr->db->d_transactions[tran_ret].end = ret;

    atomicAdd(&curr->db->transaction_tracker, 1);
    __threadfence_system();
}

__device__ void printBucketUtil(Item *local_util, int max_item)
{
    for (int i = 0; i < max_item; i++)
    {
        {
            printf("%d:%d ", local_util[i].key, local_util[i].util);
        }
    }
    printf("\n");
}

__device__ void iterative_project(WorkItem *old, WorkItem *curr, Item *local_util)
{
    // Get the item to project on.
    // (Assuming curr->pattern[0] holds a valid index or count.)
    int item = curr->pattern[curr->pattern_length - 1];

    // Iterate over all transactions in the old database.
    for (int tid = 0; tid < old->db->numTransactions; tid++)
    {
        // For easier reference.
        Transaction &oldTrans = old->db->d_transactions[tid];
        

        int idx = oldTrans.findItem(item);
        if (idx == -1) {
            continue;
        }



        // Compute the number of items after the found index.
        int items_this_trans = oldTrans.length - (idx + 1);
        curr->utility += oldTrans.utility + oldTrans.data[idx].util;


        
        if (items_this_trans <= 0) {
            continue;
        }

        // Reserve space in the current (flat) data array.
        // Record the starting index for the new transaction’s data.
        int new_start = curr->db->numItems;
        curr->db->numItems += items_this_trans;

        // Create a new transaction in curr->db.
        int tran_ret = curr->db->numTransactions;
        curr->db->numTransactions += 1;
        Transaction &newTrans = curr->db->d_transactions[tran_ret];

        // Set the new transaction’s utility.
        newTrans.utility = oldTrans.utility + oldTrans.data[idx].util;
        // Set the pointer to the beginning of its block in the flat array.
        newTrans.data = curr->db->d_data + new_start;


        // Compute total utility by summing the utilities of all suffix items.
        int total_util = newTrans.utility;
        for (int i = idx + 1; i < oldTrans.length; i++) {
            total_util += oldTrans.data[i].util;
        }

        // Copy the suffix items into the new region.
        int ret = new_start;
        for (int i = idx + 1; i < oldTrans.length; i++) {
            curr->db->d_data[ret++] = oldTrans.data[i];
            // Update bucket utility.
            add_bucket_util(local_util, old->max_item * scale, oldTrans.data[i].key, total_util);
        }
        // Record the new transaction's length.
        newTrans.length = ret - new_start;
    }
}



__device__ bool sameKey(const Database *db, int t1, int t2)
{
    const Transaction &trans1 = db->d_transactions[t1];
    const Transaction &trans2 = db->d_transactions[t2];

    if (trans1.length != trans2.length)
        return false;

    for (int i = 0; i < trans1.length; i++) {
        if (trans1.data[i].key != trans2.data[i].key)
            return false;
    }
    return true;
}


// __global__ void trim_and_merge(WorkItem *curr, Item *local_util, int *hashes, Item *subtree_util, int min_util)
// {
//     int tid = blockIdx.x;

//     int curr_loc = 0;

//     int total_tran_util = curr->db->d_transactions[tid].utility;

//     for (int i = 0; i < curr->db->d_transactions[tid].length(); i++)
//     {
//         int idx = find_item(local_util, curr->max_item * scale, curr->db->d_transactions[tid].get()[i].key);
//         // printf("TID:%d\tItem: %d\tLocal Util Idx: %d\tLocal Util: %d\n", tid, curr->db->d_transactions[tid].get()[i].key, idx, local_util[idx].util);
//         if (local_util[idx].util >= min_util)
//         {
//             // make the item be written to curr_loc in the transaction
//             curr->db->d_data[curr->db->d_transactions[tid].start + curr_loc] = curr->db->d_transactions[tid].get()[i];
//             curr_loc++;
//             total_tran_util += curr->db->d_transactions[tid].get()[i].util;
//         }
//     }

//     curr->db->d_transactions[tid].end = curr->db->d_transactions[tid].start + curr_loc;

//     int temp_util = 0;
//     // update the subtree util
//     for (int i = 0; i < curr->db->d_transactions[tid].length(); i++)
//     {
//         add_bucket_util(subtree_util, curr->max_item * scale, curr->db->d_data[curr->db->d_transactions[tid].start + i].key, total_tran_util - temp_util);
//         temp_util += curr->db->d_data[curr->db->d_transactions[tid].start + i].util;
//     }

//     // printf("Mid TID: %d\n", tid);

//     int hash_idx = items_hasher(curr->db->d_data + curr->db->d_transactions[tid].start, curr->db->d_transactions[tid].length(), curr->db->numTransactions * scale);

//     while (true)
//     {
//         int old = atomicCAS(&hashes[hash_idx], -1, tid);
//         if (old == -1)
//         {
//             break;
//         }
//         // if the slot is not empty and the key is the same, merge
//         if (sameKey(curr->db, old, tid))
//         {
//             if (tid == old)
//                 break;

//             // if lenght is same
//             if (curr->db->d_transactions[old].length() != curr->db->d_transactions[tid].length())
//             {
//                 hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
//                 continue;
//             }
//             // printf("TID:%d\tLength: %d\told tid: %d\tLength: %d\n", tid, curr->db->d_transactions[tid].length(), old, curr->db->d_transactions[old].length());

//             // printf("Merging %d and %d\n", old, tid);
//             // merge
//             for (int i = 0; i < curr->db->d_transactions[old].length(); i++)
//             {
//                 // add the utility
//                 atomicAdd(&curr->db->d_data[curr->db->d_transactions[old].start + i].util, curr->db->d_data[curr->db->d_transactions[tid].start + i].util);
//             }
//             atomicAdd(&curr->db->d_transactions[old].utility, curr->db->d_transactions[tid].utility);

//             // set this start and end to 0
//             curr->db->d_transactions[tid].start = 0;
//             curr->db->d_transactions[tid].end = 0;
//             curr->db->d_transactions[tid].utility = 0;

//             // update the transaction length
//             // curr->db->d_transactions[old].end = curr->db->d_transactions[old].start + old_loc;

//             break;
//         }
//         // if the slot is not empty and the key is not the same, find the next slot
//         hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
//         // printf("Hash Idx: %d\n", hash_idx);
//     }

//     atomicAdd(&curr->db->transaction_tracker, 1);
//     __threadfence_system();

//     return;
// }

__device__ void iterative_trim_and_merge(WorkItem *curr, 
                                           Item *local_util, 
                                           int *hashes, 
                                           Item *subtree_util, 
                                           int min_util)
{
    // Loop over each transaction in the current database.
    for (int tid = 0; tid < curr->db->numTransactions; tid++)
    {
        // Reference to the current transaction.
        Transaction &tran = curr->db->d_transactions[tid];

        // curr_loc tracks the new (trimmed) position within this transaction.
        int curr_loc = 0;
        // Start with the transaction's current utility.
        int total_tran_util = tran.utility;

        // --- Trim Phase ---
        // Iterate over each item in the transaction.
        for (int i = 0; i < tran.length; i++)
        {
            // Find the index into local_util for the key of the current item.
            int idx = find_item(local_util, curr->max_item * scale, tran.data[i].key);
            // If the local utility meets the minimum threshold...
            if (local_util[idx].util >= min_util)
            {
                // Overwrite the current transaction's data in-place.
                // (Assumes tran.data points to a modifiable array.)
                tran.data[curr_loc] = tran.data[i];
                curr_loc++;
                total_tran_util += tran.data[i].util;
            }
        }
        // Update the transaction length to reflect the trimmed items.
        tran.length = curr_loc;

        // --- Update Subtree Utility ---
        int temp_util = 0;
        for (int i = 0; i < tran.length; i++)
        {
            // add_bucket_util updates the subtree utility for the given key.
            add_bucket_util(subtree_util, curr->max_item * scale, tran.data[i].key, total_tran_util - temp_util);
            temp_util += tran.data[i].util;
        }

        // --- Hashing and Merging ---
        // Compute a hash for the trimmed transaction using its data.
        int hash_idx = items_hasher(tran.data, tran.length, curr->db->numTransactions * scale);

        while (true)
        {
            // Try to insert the current transaction index into the hash table.
            int old = atomicCAS(&hashes[hash_idx], -1, tid);
            if (old == -1)
            {
                // Insertion succeeded.
                break;
            }
            // If the slot is occupied and the transactions have the same keys...
            if (sameKey(curr->db, old, tid))
            {
                // If merging with itself, do nothing.
                if (tid == old)
                    break;

                // If the transactions have different lengths, try the next hash slot.
                if (curr->db->d_transactions[old].length != tran.length)
                {
                    hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
                    continue;
                }
                // --- Merge Phase ---
                // Merge the two transactions item-by-item.
                Transaction &oldTran = curr->db->d_transactions[old];
                for (int i = 0; i < oldTran.length; i++)
                {
                    // atomicAdd(&oldTran.data[i].util, tran.data[i].util);
                    oldTran.data[i].util += tran.data[i].util;
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
            hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
        }

        // Update the transaction tracker and ensure memory ordering.
        // atomicAdd(&curr->db->transaction_tracker, 1);
        // __threadfence_system();
    }
}


__device__ void add_pattern(WorkItem &wi, int *high_utility_patterns)
{
    int count = atomicAdd(&high_utility_patterns[0], 1);

    int idx = atomicAdd(&high_utility_patterns[1], (wi.pattern_length + 2));

    // high_utility_patterns[idx] = pattern[0];
    for (int i = 0; i < wi.pattern_length; i++)
    {
        high_utility_patterns[idx + i] = wi.pattern[i];
    }
    high_utility_patterns[idx + wi.pattern_length] = wi.utility;
}

// __global__ void mine(CudaMemoryManager *mm, AtomicWorkStack *work_queue, int min_util, int32_t *high_utility_patterns)
// {
//     WorkItem *work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));
//     WorkItem *new_work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));
//     int tid = blockIdx.x;

//     while (stack_get_work_count(work_queue) > 0)
//     {
//         memset(new_work_item, 0, sizeof(WorkItem));
//         memset(work_item, 0, sizeof(WorkItem));

//         if (!stack_pop(work_queue, work_item))
//         {
//             __threadfence_system();
//             // printf("TID: %d\t%d\n", tid, stack_get_work_count(work_queue));
//             continue;
//         }

//         printf("TID: %d\tItem: %d\n", tid, work_item->primary);

//         //     // printf("Pattern: ");
//         //     // for (int i = 0; i < work_item->pattern[0]; i++)
//         //     // {
//         //     //     printf("%d ", work_item->pattern[i + 1]);
//         //     // }
//         //     // printf("(%d)\n", work_item->primary);

//         //     // printDB(work_item->db);

//         // // pattern
//         new_work_item->pattern = reinterpret_cast<int *>(mm->malloc((work_item->pattern[0] + 2) * sizeof(int)));
//         memcpy(new_work_item->pattern, work_item->pattern, (work_item->pattern[0] + 1) * sizeof(int));
//         new_work_item->pattern[++new_work_item->pattern[0]] = work_item->primary;

//         new_work_item->db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));
//         new_work_item->db->d_data = reinterpret_cast<Item *>(mm->malloc(work_item->db->numItems * sizeof(Item)));
//         new_work_item->db->d_transactions = reinterpret_cast<Transaction *>(mm->malloc(work_item->db->numTransactions * sizeof(Transaction)));

//         new_work_item->max_item = work_item->max_item;
//         Item *local_util = reinterpret_cast<Item *>(mm->malloc(new_work_item->max_item * scale * sizeof(Item)));
//         Item *subtree_util = reinterpret_cast<Item *>(mm->malloc(new_work_item->max_item * scale * sizeof(Item)));

//         printf("TID: %d\tProject\tBlock: %d\n", tid, work_item->db->numTransactions);
//         // project<<<work_item->db->numTransactions, 1>>>(work_item, new_work_item, local_util);
//         iterative_project(work_item, new_work_item, local_util);

//         // while (new_work_item->db->transaction_tracker != work_item->db->numTransactions)
//         // {
//         //     __threadfence_system();
//         // }

//         // printDB(new_work_item->db);

//         // if utility is greater than min_util add to high utility patterns
//         if (new_work_item->utility >= min_util)
//         {
//             printf("TID:%d\tPattern Count: %d\n", tid, high_utility_patterns[0]);
//             add_pattern(new_work_item->pattern, new_work_item->utility, high_utility_patterns);
//         }

//         // printf("Number of Transactions: %d\n", new_work_item->db->numTransactions);
//         if (new_work_item->db->numTransactions == 0)
//         {
//             // printf("Freeing Memory\n");
//             mm->free(new_work_item->pattern);
//             mm->free(new_work_item->db->d_data);
//             mm->free(new_work_item->db->d_transactions);
//             mm->free(local_util);
//             mm->free(subtree_util);
//             mm->free(new_work_item->work_done);
//             mm->free(new_work_item->db);

//             atomicSub(&work_queue->active, 1);
//             int ret = atomicAdd(&work_item->work_done[0], 1);
//             if (ret == work_item->work_count - 1)
//             {

//                 mm->free(work_item->pattern);
//                 mm->free(work_item->db->d_data);
//                 mm->free(work_item->db->d_transactions);
//                 mm->free(work_item->db);
//                 mm->free(work_item->work_done);
//             }

//             continue;
//         }

//         // // trim and merge
//         new_work_item->db->transaction_tracker = 0;
//         int *hashes = reinterpret_cast<int *>(mm->malloc(new_work_item->db->numTransactions * scale * sizeof(int)));
//         memset(hashes, -1, new_work_item->db->numTransactions * sizeof(int) * scale);
//         new_work_item->work_done = reinterpret_cast<int *>(mm->malloc(sizeof(int)));

//         // printf("TID: %d\tTrim and Merge\tBlock: %d\n", tid, new_work_item->db->numTransactions);
//         // trim_and_merge<<<new_work_item->db->numTransactions, 1>>>(new_work_item, local_util, hashes, subtree_util, min_util);
//         iterative_trim_and_merge(new_work_item, local_util, hashes, subtree_util, min_util);
//         // while (new_work_item->db->transaction_tracker != new_work_item->db->numTransactions)
//         // {
//         //     // printf("TID: %d\tWaiting\tTracker: %d\tNum Transactions: %d\n", tid, new_work_item->db->transaction_tracker, new_work_item->db->numTransactions);
//         //     __threadfence();
//         // }

//         int max_item = 0;
//         // go through local util and find number of items greater than min_util
//         for (int i = 0; i < work_item->max_item * scale; i++)
//         {
//             if (local_util[i].util >= min_util)
//             {
//                 max_item++;
//             }
//         }

//         int primary_count = 0;
//         for (int i = 0; i < work_item->max_item * scale; i++)
//         {
//             if (subtree_util[i].util >= min_util)
//             {
//                 primary_count++;
//             }
//         }

//         if (primary_count == 0)
//         {
//             mm->free(new_work_item->pattern);
//             mm->free(new_work_item->db->d_data);
//             mm->free(new_work_item->db->d_transactions);
//             mm->free(local_util);
//             mm->free(subtree_util);
//             mm->free(hashes);
//             mm->free(new_work_item->work_done);
//             mm->free(new_work_item->db);

//             atomicSub(&work_queue->active, 1);
//             int ret = atomicAdd(&work_item->work_done[0], 1);
//             if (ret == work_item->work_count - 1)
//             {
//                 mm->free(work_item->pattern);
//                 mm->free(work_item->db->d_data);
//                 mm->free(work_item->db->d_transactions);
//                 mm->free(work_item->db);
//                 mm->free(work_item->work_done);
//             }

//             continue;
//         }

//         // printf("Primary Count: %d\n", primary_count);
//         // printf("Max Item: %d\n", max_item);

//         new_work_item->max_item = max_item;
//         for (int i = 0; i < work_item->max_item * scale; i++)
//         {
//             if (subtree_util[i].util >= min_util)
//             {
//                 new_work_item->primary = subtree_util[i].key;
//                 new_work_item->work_count = primary_count;
//                 stack_push(work_queue, *new_work_item);
//             }
//         }

//         // free the memory
//         mm->free(local_util);
//         mm->free(hashes);
//         mm->free(subtree_util);

//         atomicSub(&work_queue->active, 1);
//         int ret = atomicAdd(&work_item->work_done[0], 1);
//         if (ret == work_item->work_count - 1)
//         {
//             // deviceMemFree(memory_manager, work_item->base_ptr, work_item->bytes_to_alloc);
//             // mm->free(work_item->base_ptr);
//             mm->free(work_item->pattern);
//             mm->free(work_item->db->d_data);
//             mm->free(work_item->db->d_transactions);
//             mm->free(work_item->db);
//             mm->free(work_item->work_done);
//         }
//         //     // printf("\n");
//         //     // printf("Work Count: %d\n", stack_get_work_count(work_queue));
//     }

//     mm->free(work_item);
//     mm->free(new_work_item);
// }

// __global__ void new_mine(CudaMemoryManager *mm, AtomicWorkStack *curr_work_queue, AtomicWorkStack *new_work_queue, int min_util, int32_t *high_utility_patterns)
// {

//     int tid = blockIdx.x;

//     WorkItem *work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));
//     WorkItem *new_work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));
//     memset(new_work_item, 0, sizeof(WorkItem));
//     new_work_item->max_item = 0;
//     new_work_item->primary = 0;
//     new_work_item->utility = 0;
//     new_work_item->work_count = 0;

//     memset(work_item, 0, sizeof(WorkItem));


//     work_item = &curr_work_queue->items[tid];


 
//     new_work_item->pattern = reinterpret_cast<int *>(mm->malloc((work_item->pattern[0] + 2) * sizeof(int)));
//     memcpy(new_work_item->pattern, work_item->pattern, (work_item->pattern[0] + 1) * sizeof(int));
//     new_work_item->pattern[++new_work_item->pattern[0]] = work_item->primary;

//     new_work_item->db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));
//     new_work_item->db->d_data = reinterpret_cast<Item *>(mm->malloc(work_item->db->numItems * sizeof(Item)));
//     new_work_item->db->d_transactions = reinterpret_cast<Transaction *>(mm->malloc(work_item->db->numTransactions * sizeof(Transaction)));

//     memset(new_work_item->db->d_data, 0, work_item->db->numItems * sizeof(Item));
//     memset(new_work_item->db->d_transactions, 0, work_item->db->numTransactions * sizeof(Transaction));
    

//     new_work_item->max_item = work_item->max_item;
//     Item *local_util = reinterpret_cast<Item *>(mm->malloc(new_work_item->max_item * scale * sizeof(Item)));
//     Item *subtree_util = reinterpret_cast<Item *>(mm->malloc(new_work_item->max_item * scale * sizeof(Item)));

//     // iterative_project(work_item, new_work_item, local_util);

//     for (int i = 0; i < work_item->db->numTransactions; i++)
//     {
//         // find the item in old
//         int idx = work_item->db->d_transactions[i].findItem(work_item->primary);
//         printf("Line: %d\tItem: %d\tIdx: %d\tTranUtil:%d\n", i, work_item->primary, idx, work_item->db->d_transactions[i].utility);
//         if (idx == -1)
//         {
//             continue;
//         }
//     }



//     // if utility is greater than min_util add to high utility patterns
//     printf("TID: %d\tItem: %d\tUtility: %d\n", tid, work_item->primary, new_work_item->utility);
//     if (new_work_item->utility >= min_util)
//     {
//         printf("TID:%d\tPattern Count: %d\n", tid, high_utility_patterns[0]);
//         add_pattern(new_work_item->pattern, new_work_item->utility, high_utility_patterns);
//     }

//         // // printf("Number of Transactions: %d\n", new_work_item->db->numTransactions);
//         // if (new_work_item->db->numTransactions == 0)
//         // {
//         //     // printf("Freeing Memory\n");
//         //     mm->free(new_work_item->pattern);
//         //     mm->free(new_work_item->db->d_data);
//         //     mm->free(new_work_item->db->d_transactions);
//         //     mm->free(local_util);
//         //     mm->free(subtree_util);
//         //     mm->free(new_work_item->work_done);
//         //     mm->free(new_work_item->db);

//         //     atomicSub(&work_queue->active, 1);
//         //     int ret = atomicAdd(&work_item->work_done[0], 1);
//         //     if (ret == work_item->work_count - 1)
//         //     {

//         //         mm->free(work_item->pattern);
//         //         mm->free(work_item->db->d_data);
//         //         mm->free(work_item->db->d_transactions);
//         //         mm->free(work_item->db);
//         //         mm->free(work_item->work_done);
//         //     }

//         //     continue;
//         // }

//         // // // trim and merge
//         // new_work_item->db->transaction_tracker = 0;
//         // int *hashes = reinterpret_cast<int *>(mm->malloc(new_work_item->db->numTransactions * scale * sizeof(int)));
//         // memset(hashes, -1, new_work_item->db->numTransactions * sizeof(int) * scale);
//         // new_work_item->work_done = reinterpret_cast<int *>(mm->malloc(sizeof(int)));

//         // // printf("TID: %d\tTrim and Merge\tBlock: %d\n", tid, new_work_item->db->numTransactions);
//         // // trim_and_merge<<<new_work_item->db->numTransactions, 1>>>(new_work_item, local_util, hashes, subtree_util, min_util);
//         // iterative_trim_and_merge(new_work_item, local_util, hashes, subtree_util, min_util);
//         // // while (new_work_item->db->transaction_tracker != new_work_item->db->numTransactions)
//         // // {
//         // //     // printf("TID: %d\tWaiting\tTracker: %d\tNum Transactions: %d\n", tid, new_work_item->db->transaction_tracker, new_work_item->db->numTransactions);
//         // //     __threadfence();
//         // // }

//         // int max_item = 0;
//         // // go through local util and find number of items greater than min_util
//         // for (int i = 0; i < work_item->max_item * scale; i++)
//         // {
//         //     if (local_util[i].util >= min_util)
//         //     {
//         //         max_item++;
//         //     }
//         // }

//         // int primary_count = 0;
//         // for (int i = 0; i < work_item->max_item * scale; i++)
//         // {
//         //     if (subtree_util[i].util >= min_util)
//         //     {
//         //         primary_count++;
//         //     }
//         // }

//         // if (primary_count == 0)
//         // {
//         //     mm->free(new_work_item->pattern);
//         //     mm->free(new_work_item->db->d_data);
//         //     mm->free(new_work_item->db->d_transactions);
//         //     mm->free(local_util);
//         //     mm->free(subtree_util);
//         //     mm->free(hashes);
//         //     mm->free(new_work_item->work_done);
//         //     mm->free(new_work_item->db);

//         //     atomicSub(&work_queue->active, 1);
//         //     int ret = atomicAdd(&work_item->work_done[0], 1);
//         //     if (ret == work_item->work_count - 1)
//         //     {
//         //         mm->free(work_item->pattern);
//         //         mm->free(work_item->db->d_data);
//         //         mm->free(work_item->db->d_transactions);
//         //         mm->free(work_item->db);
//         //         mm->free(work_item->work_done);
//         //     }

//         //     continue;
//         // }

//         // // printf("Primary Count: %d\n", primary_count);
//         // // printf("Max Item: %d\n", max_item);

//         // new_work_item->max_item = max_item;
//         // for (int i = 0; i < work_item->max_item * scale; i++)
//         // {
//         //     if (subtree_util[i].util >= min_util)
//         //     {
//         //         new_work_item->primary = subtree_util[i].key;
//         //         new_work_item->work_count = primary_count;
//         //         stack_push(work_queue, *new_work_item);
//         //     }
//         // }

//         // // free the memory
//         // mm->free(local_util);
//         // mm->free(hashes);
//         // mm->free(subtree_util);

//         // atomicSub(&work_queue->active, 1);
//         // int ret = atomicAdd(&work_item->work_done[0], 1);
//         // if (ret == work_item->work_count - 1)
//         // {
//         //     // deviceMemFree(memory_manager, work_item->base_ptr, work_item->bytes_to_alloc);
//         //     // mm->free(work_item->base_ptr);
//         //     mm->free(work_item->pattern);
//         //     mm->free(work_item->db->d_data);
//         //     mm->free(work_item->db->d_transactions);
//         //     mm->free(work_item->db);
//         //     mm->free(work_item->work_done);
//         // }
//         // //     // printf("\n");
//         // //     // printf("Work Count: %d\n", stack_get_work_count(work_queue));


// }


// template <typename MemoryManagerType>
__global__ void test(AtomicWorkStack *curr_work_queue, AtomicWorkStack *new_work_queue, int32_t *d_high_utility_patterns, CudaMemoryManager *mm, int min_util)
{

    int i = blockIdx.x;
    // printf("\nNumber of current work items: %d\n", curr_work_queue->top);
    // for (int i = 0; i < curr_work_queue->top; i++)
    {
        WorkItem item = curr_work_queue->items[i];


        // for (int j = 0; j < item.pattern_length; j++)
        // {
        //     printf("%d ", item.pattern[j]);
        // }
        // printf("(%d)\n", item.primary);

        // printDatabase(item.db);
        


        WorkItem new_work_item;
        memset(&new_work_item, 0, sizeof(WorkItem)); // Initialize the new work item

        new_work_item.pattern = reinterpret_cast<int *>(mm->malloc((item.pattern_length + 1) * sizeof(int)));
        memcpy(new_work_item.pattern, item.pattern, (item.pattern_length) * sizeof(int));
        new_work_item.pattern[item.pattern_length] = item.primary;
        new_work_item.pattern_length = item.pattern_length + 1;
        
        // printf("Pattern: ");
        // for (int i = 0; i < new_work_item.pattern_length; i++)
        // {
        //     printf("%d ", new_work_item.pattern[i]);
        // }
        // printf("\n");


        new_work_item.db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));
        memset(new_work_item.db, 0, sizeof(Database));
        new_work_item.db->d_data = reinterpret_cast<Item *>(mm->malloc(item.db->numItems * sizeof(Item)));
        new_work_item.db->d_transactions = reinterpret_cast<Transaction *>(mm->malloc(item.db->numTransactions * sizeof(Transaction)));

        memset(new_work_item.db->d_data, 0, item.db->numItems * sizeof(Item));
        memset(new_work_item.db->d_transactions, 0, item.db->numTransactions * sizeof(Transaction));
        

        new_work_item.max_item = item.max_item;
        Item *local_util = reinterpret_cast<Item *>(mm->malloc(new_work_item.max_item * scale * sizeof(Item)));
        memset(local_util, 0, new_work_item.max_item * scale * sizeof(Item));
        Item *subtree_util = reinterpret_cast<Item *>(mm->malloc(new_work_item.max_item * scale * sizeof(Item)));
        memset(subtree_util, 0, new_work_item.max_item * scale * sizeof(Item));


        iterative_project(&item, &new_work_item, local_util);

        printf("Local Util: ");
        printBucketUtil(local_util, item.max_item * scale);

        printDatabase(new_work_item.db);


        for (int j = 0; j < new_work_item.pattern_length; j++)
        {
            printf("%d ", new_work_item.pattern[j]);
        }
        printf(": %d\n", new_work_item.utility);


        // if utility is greater than min_util add to high utility patterns
        if (new_work_item.utility >= min_util)
        {
            add_pattern(new_work_item, d_high_utility_patterns);
        }

//         // // printf("Number of Transactions: %d\n", new_work_item->db->numTransactions);
        printf("Number of Transactions: %d\n", new_work_item.db->numTransactions);

        if (new_work_item.db->numTransactions == 0)
        {
            // printf("Freeing Memory\n");
            mm->free(new_work_item.pattern);
            mm->free(new_work_item.db->d_data);
            mm->free(new_work_item.db->d_transactions);
            mm->free(local_util);
            mm->free(subtree_util);
            mm->free(new_work_item.work_done);
            mm->free(new_work_item.db);

            // atomicSub(&curr.active, 1);
            int ret = atomicAdd(&item.work_done[0], 1);
            // printf("Work Count: %d\n", item.work_count);
            if (ret == (item.work_count- 1))
            {
                // printf("Freeing Memory\n");
                mm->free(item.pattern);
                mm->free(item.db->d_data);
                mm->free(item.db->d_transactions);
                mm->free(item.db);
                mm->free(item.work_done);
            }

            // continue;
            return;
        }

// //         // // // trim and merge
//         // new_work_item->db->transaction_tracker = 0;
        int *hashes = (int *)(mm->malloc(new_work_item.db->numTransactions * scale * sizeof(int)));
        memset(hashes, -1, new_work_item.db->numTransactions * sizeof(int) * scale);
        new_work_item.work_done = reinterpret_cast<int *>(mm->malloc(sizeof(int)));

// //         // // printf("TID: %d\tTrim and Merge\tBlock: %d\n", tid, new_work_item->db->numTransactions);
// //         // // trim_and_merge<<<new_work_item->db->numTransactions, 1>>>(new_work_item, local_util, hashes, subtree_util, min_util);
        iterative_trim_and_merge(&new_work_item, local_util, hashes, subtree_util, min_util);
        // printf("Trim and Merge\n");
        // printDatabase(new_work_item.db);

        // printf("Subtree Util: ");
        // printBucketUtil(subtree_util, item.max_item * scale);

// //         // // while (new_work_item->db->transaction_tracker != new_work_item->db->numTransactions)
// //         // // {
// //         // //     // printf("TID: %d\tWaiting\tTracker: %d\tNum Transactions: %d\n", tid, new_work_item->db->transaction_tracker, new_work_item->db->numTransactions);
// //         // //     __threadfence();
// //         // // }

        int max_item = 0;
        // go through local util and find number of items greater than min_util
        for (int i = 0; i < item.max_item * scale; i++)
        {
            if (local_util[i].util >= min_util)
            {
                max_item++;
            }
        }

        int primary_count = 0;
        for (int i = 0; i < item.max_item * scale; i++)
        {
            if (subtree_util[i].util >= min_util)
            {
                primary_count++;
            }
        }
        // printf("Primary Count: %d\n", primary_count);
        // printf("Max Item: %d\n", max_item);


        if (primary_count == 0)
        {

            mm->free(new_work_item.pattern);
            mm->free(new_work_item.db->d_data);
            mm->free(new_work_item.db->d_transactions);
            mm->free(local_util);
            mm->free(subtree_util);
            mm->free(hashes);
            mm->free(new_work_item.work_done);
            mm->free(new_work_item.db);
            
//         //     atomicSub(&work_queue->active, 1);
            int ret = atomicAdd(&item.work_done[0], 1);
            if (ret == item.work_count - 1)
            {
                mm->free(item.pattern);
                mm->free(item.db->d_data);
                mm->free(item.db->d_transactions);
                mm->free(item.db);
                mm->free(item.work_done);
            }

            // continue;
            return;

        }

// //         // // printf("Primary Count: %d\n", primary_count);
// //         // // printf("Max Item: %d\n", max_item);

        new_work_item.max_item = max_item;
        for (int i = 0; i < item.max_item * scale; i++)
        {
            if (subtree_util[i].util >= min_util)
            {
                new_work_item.primary = subtree_util[i].key;
                new_work_item.work_count = primary_count;
                // stack_push(work_queue, *new_work_item);
                new_work_queue->push(new_work_item);
            }
        }

        // free the memory
        mm->free(local_util);
        mm->free(hashes);
        mm->free(subtree_util);

// //         // atomicSub(&work_queue->active, 1);
// //         // int ret = atomicAdd(&work_item->work_done[0], 1);
// //         // if (ret == work_item->work_count - 1)
// //         // {
// //         //     // deviceMemFree(memory_manager, work_item->base_ptr, work_item->bytes_to_alloc);
// //         //     // mm->free(work_item->base_ptr);
// //         //     mm->free(work_item->pattern);
// //         //     mm->free(work_item->db->d_data);
// //         //     mm->free(work_item->db->d_transactions);
// //         //     mm->free(work_item->db);
// //         //     mm->free(work_item->work_done);
// //         // }
// //         // //     // printf("\n");
// //         // //     // printf("Work Count: %d\n", stack_get_work_count(work_queue));


// // }
        // printf("\n");
// return;

    }

   
}