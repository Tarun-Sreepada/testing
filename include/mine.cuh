#pragma once

#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager
#include "mine.cuh"

#define scale 2

__global__ void copy(
    CudaMemoryManager *mm,
    AtomicWorkStack *work_queue,
    Item *items,
    int *start,
    int *end,
    int *primary,
    int num_primary,
    int num_transactions,
    int max_item)
{

    stack_init(work_queue);

    int item_count = end[num_transactions - 1] - start[0];

    WorkItem work_item;

    // pattern
    work_item.pattern = reinterpret_cast<int *>(mm->malloc((2 + num_primary) * sizeof(int)));

    // db
    work_item.db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));

    // items
    // work_item.db->d_data = reinterpret_cast<Item *>(base_ptr);
    work_item.db->d_data = reinterpret_cast<Item *>(mm->malloc(item_count * sizeof(Item)));
    memcpy(work_item.db->d_data, items, item_count * sizeof(Item));
    work_item.db->numItems = item_count;

    // transactions
    work_item.db->d_transactions = reinterpret_cast<Transaction *>(mm->malloc(num_transactions * sizeof(Transaction)));
    work_item.db->numTransactions = num_transactions;
    for (int i = 0; i < num_transactions; i++)
    {
        work_item.db->d_transactions[i].data = work_item.db->d_data;
        work_item.db->d_transactions[i].utility = 0;
        work_item.db->d_transactions[i].start = start[i];
        work_item.db->d_transactions[i].end = end[i];
    }
    // counts
    work_item.db->numItems = item_count;
    work_item.db->numTransactions = num_transactions;

    work_item.work_done = reinterpret_cast<int *>(mm->malloc(sizeof(int)));
    work_item.work_count = num_primary;
    work_item.max_item = max_item;

    for (int i = 0; i < num_primary; i++)
    {
        work_item.primary = primary[i];

        stack_push(work_queue, work_item);
    }
}

__device__ void printDB(Database *db)
{
    printf("DB: \n");
    for (int i = 0; i < db->numTransactions; i++)
    {
        printf("%d|", db->d_transactions[i].utility);
        for (int j = 0; j < db->d_transactions[i].length(); j++)
        {
            printf("%d:%d ", db->d_transactions[i].get()[j].key, db->d_transactions[i].get()[j].util);
        }
        printf("\n");
    }
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

__device__ void iterative_project(WorkItem *old, WorkItem *curr, Item *local_util)
{

    // int tid = blockIdx.x;

    for (int tid = 0; tid < old->db->numTransactions; tid++)
    {
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
            continue;

            // return;
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
    // printf("Tracker: %d\n", curr->db->transaction_tracker);
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

__device__ bool sameKey(const Database *db, int t1, int t2)
{
    int len1 = db->d_transactions[t1].length();
    int len2 = db->d_transactions[t2].length();
    if (len1 != len2)
        return false;

    int start1 = db->d_transactions[t1].start;
    int start2 = db->d_transactions[t2].start;

    // Compare each item
    for (int i = 0; i < len1; i++)
    {
        // If you store item IDs in d_data[...].id, for example:
        if (db->d_data[start1 + i].key != db->d_data[start2 + i].key)
        {
            return false;
        }
    }
    return true;
}

__global__ void trim_and_merge(WorkItem *curr, Item *local_util, int *hashes, Item *subtree_util, int min_util)
{
    int tid = blockIdx.x;

    int curr_loc = 0;

    int total_tran_util = curr->db->d_transactions[tid].utility;

    for (int i = 0; i < curr->db->d_transactions[tid].length(); i++)
    {
        int idx = find_item(local_util, curr->max_item * scale, curr->db->d_transactions[tid].get()[i].key);
        // printf("TID:%d\tItem: %d\tLocal Util Idx: %d\tLocal Util: %d\n", tid, curr->db->d_transactions[tid].get()[i].key, idx, local_util[idx].util);
        if (local_util[idx].util >= min_util)
        {
            // make the item be written to curr_loc in the transaction
            curr->db->d_data[curr->db->d_transactions[tid].start + curr_loc] = curr->db->d_transactions[tid].get()[i];
            curr_loc++;
            total_tran_util += curr->db->d_transactions[tid].get()[i].util;
        }
    }

    curr->db->d_transactions[tid].end = curr->db->d_transactions[tid].start + curr_loc;

    int temp_util = 0;
    // update the subtree util
    for (int i = 0; i < curr->db->d_transactions[tid].length(); i++)
    {
        add_bucket_util(subtree_util, curr->max_item * scale, curr->db->d_data[curr->db->d_transactions[tid].start + i].key, total_tran_util - temp_util);
        temp_util += curr->db->d_data[curr->db->d_transactions[tid].start + i].util;
    }

    // printf("Mid TID: %d\n", tid);

    int hash_idx = items_hasher(curr->db->d_data + curr->db->d_transactions[tid].start, curr->db->d_transactions[tid].length(), curr->db->numTransactions * scale);

    while (true)
    {
        int old = atomicCAS(&hashes[hash_idx], -1, tid);
        if (old == -1)
        {
            break;
        }
        // if the slot is not empty and the key is the same, merge
        if (sameKey(curr->db, old, tid))
        {
            if (tid == old)
                break;

            // if lenght is same
            if (curr->db->d_transactions[old].length() != curr->db->d_transactions[tid].length())
            {
                hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
                continue;
            }
            // printf("TID:%d\tLength: %d\told tid: %d\tLength: %d\n", tid, curr->db->d_transactions[tid].length(), old, curr->db->d_transactions[old].length());

            // printf("Merging %d and %d\n", old, tid);
            // merge
            for (int i = 0; i < curr->db->d_transactions[old].length(); i++)
            {
                // add the utility
                atomicAdd(&curr->db->d_data[curr->db->d_transactions[old].start + i].util, curr->db->d_data[curr->db->d_transactions[tid].start + i].util);
            }
            atomicAdd(&curr->db->d_transactions[old].utility, curr->db->d_transactions[tid].utility);

            // set this start and end to 0
            curr->db->d_transactions[tid].start = 0;
            curr->db->d_transactions[tid].end = 0;
            curr->db->d_transactions[tid].utility = 0;

            // update the transaction length
            // curr->db->d_transactions[old].end = curr->db->d_transactions[old].start + old_loc;

            break;
        }
        // if the slot is not empty and the key is not the same, find the next slot
        hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
        // printf("Hash Idx: %d\n", hash_idx);
    }

    atomicAdd(&curr->db->transaction_tracker, 1);
    __threadfence_system();

    return;
}

__device__ void iterative_trim_and_merge(WorkItem *curr, Item *local_util, int *hashes, Item *subtree_util, int min_util)
{
    // int tid = blockIdx.x;
    for (int tid = 0; tid < curr->db->numTransactions; tid++)
    {

    int curr_loc = 0;

    int total_tran_util = curr->db->d_transactions[tid].utility;

    for (int i = 0; i < curr->db->d_transactions[tid].length(); i++)
    {
        int idx = find_item(local_util, curr->max_item * scale, curr->db->d_transactions[tid].get()[i].key);
        // printf("TID:%d\tItem: %d\tLocal Util Idx: %d\tLocal Util: %d\n", tid, curr->db->d_transactions[tid].get()[i].key, idx, local_util[idx].util);
        if (local_util[idx].util >= min_util)
        {
            // make the item be written to curr_loc in the transaction
            curr->db->d_data[curr->db->d_transactions[tid].start + curr_loc] = curr->db->d_transactions[tid].get()[i];
            curr_loc++;
            total_tran_util += curr->db->d_transactions[tid].get()[i].util;
        }
    }

    curr->db->d_transactions[tid].end = curr->db->d_transactions[tid].start + curr_loc;

    int temp_util = 0;
    // update the subtree util
    for (int i = 0; i < curr->db->d_transactions[tid].length(); i++)
    {
        add_bucket_util(subtree_util, curr->max_item * scale, curr->db->d_data[curr->db->d_transactions[tid].start + i].key, total_tran_util - temp_util);
        temp_util += curr->db->d_data[curr->db->d_transactions[tid].start + i].util;
    }

    // printf("Mid TID: %d\n", tid);

    int hash_idx = items_hasher(curr->db->d_data + curr->db->d_transactions[tid].start, curr->db->d_transactions[tid].length(), curr->db->numTransactions * scale);

    while (true)
    {
        int old = atomicCAS(&hashes[hash_idx], -1, tid);
        if (old == -1)
        {
            break;
        }
        // if the slot is not empty and the key is the same, merge
        if (sameKey(curr->db, old, tid))
        {
            if (tid == old)
                break;

            // if lenght is same
            if (curr->db->d_transactions[old].length() != curr->db->d_transactions[tid].length())
            {
                hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
                continue;
            }
            // printf("TID:%d\tLength: %d\told tid: %d\tLength: %d\n", tid, curr->db->d_transactions[tid].length(), old, curr->db->d_transactions[old].length());

            // printf("Merging %d and %d\n", old, tid);
            // merge
            for (int i = 0; i < curr->db->d_transactions[old].length(); i++)
            {
                // add the utility
                atomicAdd(&curr->db->d_data[curr->db->d_transactions[old].start + i].util, curr->db->d_data[curr->db->d_transactions[tid].start + i].util);
            }
            atomicAdd(&curr->db->d_transactions[old].utility, curr->db->d_transactions[tid].utility);

            // set this start and end to 0
            curr->db->d_transactions[tid].start = 0;
            curr->db->d_transactions[tid].end = 0;
            curr->db->d_transactions[tid].utility = 0;

            // update the transaction length
            // curr->db->d_transactions[old].end = curr->db->d_transactions[old].start + old_loc;

            break;
        }
        // if the slot is not empty and the key is not the same, find the next slot
        hash_idx = (hash_idx + 1) % (curr->db->numTransactions * scale);
        // printf("Hash Idx: %d\n", hash_idx);
    }

    atomicAdd(&curr->db->transaction_tracker, 1);
    __threadfence_system();

    // return;
    }
}

__device__ void add_pattern(int *pattern, int utility, int *high_utility_patterns)
{
    int count = atomicAdd(&high_utility_patterns[0], 1);

    int idx = atomicAdd(&high_utility_patterns[1], pattern[0] + 2);

    // high_utility_patterns[idx] = pattern[0];
    for (int i = 0; i < pattern[0]; i++)
    {
        high_utility_patterns[idx + i] = pattern[i + 1];
    }
    high_utility_patterns[idx + pattern[0]] = utility;
}

__global__ void mine(CudaMemoryManager *mm, AtomicWorkStack *work_queue, int min_util, int32_t *high_utility_patterns)
{
    WorkItem *work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));
    WorkItem *new_work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));
    int tid = blockIdx.x;

    while (stack_get_work_count(work_queue) > 0)
    {
        memset(new_work_item, 0, sizeof(WorkItem));
        memset(work_item, 0, sizeof(WorkItem));

        if (!stack_pop(work_queue, work_item))
        {
            __threadfence_system();
            continue;
        }

        printf("TID: %d\tItem: %d\n", tid, work_item->primary);

        // printf("Pattern: ");
        // for (int i = 0; i < work_item->pattern[0]; i++)
        // {
        //     printf("%d ", work_item->pattern[i + 1]);
        // }
        // printf("(%d)\n", work_item->primary);

        // printDB(work_item->db);

        // // pattern
        new_work_item->pattern = reinterpret_cast<int *>(mm->malloc((work_item->pattern[0] + 2) * sizeof(int)));
        memcpy(new_work_item->pattern, work_item->pattern, (work_item->pattern[0] + 1) * sizeof(int));
        new_work_item->pattern[++new_work_item->pattern[0]] = work_item->primary;

        new_work_item->db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));
        new_work_item->db->d_data = reinterpret_cast<Item *>(mm->malloc(work_item->db->numItems * sizeof(Item)));
        new_work_item->db->d_transactions = reinterpret_cast<Transaction *>(mm->malloc(work_item->db->numTransactions * sizeof(Transaction)));

        new_work_item->max_item = work_item->max_item;
        Item *local_util = reinterpret_cast<Item *>(mm->malloc(new_work_item->max_item * scale * sizeof(Item)));
        Item *subtree_util = reinterpret_cast<Item *>(mm->malloc(new_work_item->max_item * scale * sizeof(Item)));

        printf("TID: %d\tProject\tBlock: %d\n", tid, work_item->db->numTransactions);
        // project<<<work_item->db->numTransactions, 1>>>(work_item, new_work_item, local_util);
        iterative_project(work_item, new_work_item, local_util);
            printf("TID: %d\tWaiting\tTracker: %d\tNum Transactions: %d\n", tid, new_work_item->db->transaction_tracker, work_item->db->numTransactions);


        while (new_work_item->db->transaction_tracker != work_item->db->numTransactions)
        {
            __threadfence_system();
        }

        // printDB(new_work_item->db);

        // if utility is greater than min_util add to high utility patterns
        if (new_work_item->utility >= min_util)
        {
            printf("TID:%d\tPattern Count: %d\n", tid, high_utility_patterns[0]);
            add_pattern(new_work_item->pattern, new_work_item->utility, high_utility_patterns);
        }

        // printf("Number of Transactions: %d\n", new_work_item->db->numTransactions);
        if (new_work_item->db->numTransactions == 0)
        {
            // printf("Freeing Memory\n");
            mm->free(new_work_item->pattern);
            mm->free(new_work_item->db->d_data);
            mm->free(new_work_item->db->d_transactions);
            mm->free(local_util);
            mm->free(subtree_util);
            mm->free(new_work_item->work_done);
            mm->free(new_work_item->db);

            atomicSub(&work_queue->active, 1);
            int ret = atomicAdd(&work_item->work_done[0], 1);
            if (ret == work_item->work_count - 1)
            {

                mm->free(work_item->pattern);
                mm->free(work_item->db->d_data);
                mm->free(work_item->db->d_transactions);
                mm->free(work_item->db);
                mm->free(work_item->work_done);
            }

            continue;
        }

        // trim and merge
        new_work_item->db->transaction_tracker = 0;
        int *hashes = reinterpret_cast<int *>(mm->malloc(new_work_item->db->numTransactions * scale * sizeof(int)));
        memset(hashes, -1, new_work_item->db->numTransactions * sizeof(int) * scale);
        new_work_item->work_done = reinterpret_cast<int *>(mm->malloc(sizeof(int)));

        printf("TID: %d\tTrim and Merge\tBlock: %d\n", tid, new_work_item->db->numTransactions);
        // trim_and_merge<<<new_work_item->db->numTransactions, 1>>>(new_work_item, local_util, hashes, subtree_util, min_util);
        iterative_trim_and_merge(new_work_item, local_util, hashes, subtree_util, min_util);
        while (new_work_item->db->transaction_tracker != new_work_item->db->numTransactions)
        {
            // printf("TID: %d\tWaiting\tTracker: %d\tNum Transactions: %d\n", tid, new_work_item->db->transaction_tracker, new_work_item->db->numTransactions);
            __threadfence();
        }

        int max_item = 0;
        // go through local util and find number of items greater than min_util
        for (int i = 0; i < work_item->max_item * scale; i++)
        {
            if (local_util[i].util >= min_util)
            {
                max_item++;
            }
        }

        int primary_count = 0;
        for (int i = 0; i < work_item->max_item * scale; i++)
        {
            if (subtree_util[i].util >= min_util)
            {
                primary_count++;
            }
        }

        if (primary_count == 0)
        {
            mm->free(new_work_item->pattern);
            mm->free(new_work_item->db->d_data);
            mm->free(new_work_item->db->d_transactions);
            mm->free(local_util);
            mm->free(subtree_util);
            mm->free(hashes);
            mm->free(new_work_item->work_done);
            mm->free(new_work_item->db);

            atomicSub(&work_queue->active, 1);
            int ret = atomicAdd(&work_item->work_done[0], 1);
            if (ret == work_item->work_count - 1)
            {
                mm->free(work_item->pattern);
                mm->free(work_item->db->d_data);
                mm->free(work_item->db->d_transactions);
                mm->free(work_item->db);
                mm->free(work_item->work_done);
            }

            continue;
        }

        // printf("Primary Count: %d\n", primary_count);
        // printf("Max Item: %d\n", max_item);

        new_work_item->max_item = max_item;
        for (int i = 0; i < work_item->max_item * scale; i++)
        {
            if (subtree_util[i].util >= min_util)
            {
                new_work_item->primary = subtree_util[i].key;
                new_work_item->work_count = primary_count;
                stack_push(work_queue, *new_work_item);
            }
        }

        // free the memory
        mm->free(local_util);
        mm->free(hashes);
        mm->free(subtree_util);

        atomicSub(&work_queue->active, 1);
        int ret = atomicAdd(&work_item->work_done[0], 1);
        if (ret == work_item->work_count - 1)
        {
            // deviceMemFree(memory_manager, work_item->base_ptr, work_item->bytes_to_alloc);
            // mm->free(work_item->base_ptr);
            mm->free(work_item->pattern);
            mm->free(work_item->db->d_data);
            mm->free(work_item->db->d_transactions);
            mm->free(work_item->db);
            mm->free(work_item->work_done);
        }
        // printf("\n");
        // printf("Work Count: %d\n", stack_get_work_count(work_queue));
    }

    mm->free(work_item);
    mm->free(new_work_item);
}