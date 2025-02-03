
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define scale 2
#define page_size 512
#define total_memory 6 * GIGA

#define blocks 1

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
    // printf("\n\n");
}


// __device__ void add_local_util(local_util, old->max_item * scale, old->db->d_data[i].key, total_util);
__device__ void add_bucket_util(Item *local_util, int max_item, int key, int total_util)
{
    // hash the key
    int idx = hashFunction(key, max_item * scale);

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
            // if the key is not present, add the key and utility
            atomicExch(&local_util[idx].key, key);
            atomicAdd(&local_util[idx].util, total_util);
            return;
        }
        // if the key is not present, find the next slot
        idx = (idx + 1) % (max_item * scale);
    }


}


__global__ void project(WorkItem *old, WorkItem *curr, Item *local_util)
{

    int tid = blockIdx.x;

    if (tid >= old->db->numTransactions) return;

    // find the item in old
    int item = curr->pattern[curr->pattern[0]];
    int idx = old->db->d_transactions[tid].findItem(item);
    if (idx == -1)
    {
        atomicAdd(&curr->db->transaction_tracker, 1);

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
    for (int i = idx+1; i < old->db->d_transactions[tid].end; i++)
    {
        total_util += old->db->d_data[i].util;
    }

    for (int i = idx+1; i < old->db->d_transactions[tid].end; i++)
    {
    //     ret++;
        curr->db->d_data[ret++] = old->db->d_data[i];
        add_bucket_util(local_util, old->max_item * scale, old->db->d_data[i].key, total_util);
    }
    curr->db->d_transactions[tran_ret].end = ret;

    atomicAdd(&curr->db->transaction_tracker, 1);
}

__device__ void printBucketUtil(Item *local_util, int max_item)
{
    printf("Bucket Util: \t");
    for (int i = 0; i < max_item; i++)
    {
        // if (local_util[i].key != 0)
        {
            printf("%d:%d ", local_util[i].key, local_util[i].util);
        }
    }
    printf("\n");
}


__global__ void trim_and_merge(WorkItem *curr, Item *local_util, Item *hashes, Item *subtree_util, int min_util)
{
    int tid = blockIdx.x;

    if (tid >= curr->db->numTransactions) return;
    int curr_loc = 0;

    for (int i = 0; i < curr->db->d_transactions[tid].length(); i++)
    {
        int idx = hashFunction(curr->db->d_transactions[tid].get()[i].key, curr->max_item * scale);
        printf("TID:%d\tItem: %d\tLocal Util Idx: %d\tLocal Util: %d\n", tid, curr->db->d_transactions[tid].get()[i].key, idx, local_util[idx].util);
        if (local_util[idx].util >= min_util)
        {
            // make the item be written to curr_loc in the transaction
            curr->db->d_data[curr->db->d_transactions[tid].start + curr_loc] = curr->db->d_transactions[tid].get()[i];
            curr_loc++;
        }


    }

    // update the transaction length
    curr->db->d_transactions[tid].end = curr->db->d_transactions[tid].start + curr_loc;
    printf("Transaction Length: %d\n", curr->db->d_transactions[tid].length());


    atomicAdd(&curr->db->transaction_tracker, 1);
}

__global__ void verify(CudaMemoryManager *mm, AtomicWorkStack *work_queue, int min_util)
{
    WorkItem *work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));
    printf("Work Count: %d\n", stack_get_work_count(work_queue));

    while (stack_get_work_count(work_queue) > 0)
    {
        stack_pop(work_queue, work_item);

        printf("Pattern: ");
        for (int i = 0; i < work_item->pattern[0]; i++)
        {
            printf("%d ", work_item->pattern[i + 1]);
        }
        printf("(%d)\n", work_item->primary);

        printDB(work_item->db);


        WorkItem *new_work_item = reinterpret_cast<WorkItem *>(mm->malloc(sizeof(WorkItem)));


        // pattern
        new_work_item->pattern = reinterpret_cast<int *>(mm->malloc((work_item->pattern[0] + 2) * sizeof(int)));
        memcpy(new_work_item->pattern, work_item->pattern, (work_item->pattern[0]) * sizeof(int));
        new_work_item->pattern[++new_work_item->pattern[0]] = work_item->primary;
        // base_ptr += (work_item->pattern[0] + 2) * sizeof(int);

        // db
        new_work_item->db = reinterpret_cast<Database *>(mm->malloc(sizeof(Database)));


        new_work_item->db->d_data = reinterpret_cast<Item *>(mm->malloc(work_item->db->numItems * sizeof(Item)));


        new_work_item->db->d_transactions = reinterpret_cast<Transaction *>(mm->malloc(work_item->db->numTransactions * sizeof(Transaction)));


        // Item *local_util = reinterpret_cast<Item *>(deviceMemMalloc(memory_manager, work_item->max_item * scale * sizeof(Item)));
        Item *local_util = reinterpret_cast<Item *>(mm->malloc(work_item->max_item * scale * sizeof(Item)));


        project<<<work_item->db->numTransactions, 1>>>(work_item, new_work_item, local_util);

        while (new_work_item->db->transaction_tracker != work_item->db->numTransactions)
        {
            __threadfence();
        }
        printf("Number of Transactions: %d\n", new_work_item->db->numTransactions);
        printf("Number of Items: %d\n", new_work_item->db->numItems);

        printDB(new_work_item->db);
        printBucketUtil(local_util, work_item->max_item * scale);

        if (new_work_item->db->numTransactions == 0)
        {
            mm->free(new_work_item->pattern);
            mm->free(new_work_item->db->d_data);
            mm->free(new_work_item->db->d_transactions);
            mm->free(new_work_item->db);

            mm->free(local_util);
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
        Item *hashes = reinterpret_cast<Item *>(mm->malloc(new_work_item->db->numTransactions * sizeof(Item) * scale));
        Item *subtree_util = reinterpret_cast<Item *>(mm->malloc(work_item->max_item * scale * sizeof(Item)));
        new_work_item->db->transaction_tracker = 0;
        new_work_item->max_item = work_item->max_item;

        trim_and_merge<<<new_work_item->db->numTransactions, 1>>>(new_work_item, local_util, hashes, subtree_util, min_util);

        while (new_work_item->db->transaction_tracker != new_work_item->db->numTransactions)
        {
            __threadfence();
        }

        printDB(new_work_item->db);

  
        // // print new work item
        // printf("Pattern: (len:%d) | ", new_work_item.pattern[0]);
        // for (int i = 0; i < new_work_item.pattern[0]; i++)
        // {
        //     printf("%d ", new_work_item.pattern[i + 1]);
        // }

        // printf("\n");

        // printf("Primary: %d\n", new_work_item.primary);
        // printf("Work Done: %d\n", new_work_item.work_done[0]);
        // printf("Work Count: %d|", new_work_item.work_count);
        // printf("Bytes: %d\n", new_work_item.bytes_to_alloc);
        // printf("DB: \n");
        // for (int i = 0; i < new_work_item.db->numTransactions; i++)
        // {
        //     printf("%d|", new_work_item.db->d_transactions[i].utility);
        //     for (int j = 0; j < new_work_item.db->d_transactions[i].length(); j++)
        //     {
        //         printf("%d:%d ", new_work_item.db->d_transactions[i].get()[j].key, new_work_item.db->d_transactions[i].get()[j].util);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // new_work_item.db = reinterpret_cast<Database *>(base_ptr);
        // memcpy(new_work_item.db, work_item.db, sizeof(Database));

        // // Local Util, Subtree Util
        // int bytes_for_util = 2 * work_item->max_item * scale * sizeof(Item);
        // void *n_base_ptr = deviceMemMalloc(memory_manager, bytes_for_util);

        // Item *local_util = reinterpret_cast<Item *>(n_base_ptr);
        // n_base_ptr += bytes_for_util;

        // Item *subtree_util = reinterpret_cast<Item *>(n_base_ptr);
        // n_base_ptr += bytes_for_util;

        // // Tran Hash
        // Item *tran_hash = reinterpret_cast<Item *>(deviceMemMalloc(memory_manager, work_item->db->numTransactions * sizeof(Item) * scale));

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
        printf("\n");
        printf("Work Count: %d\n", stack_get_work_count(work_queue));
    }
}

__global__ void mine(CudaMemoryManager *memory_manager, AtomicWorkStack *work_queue, int utility, int32_t *high_utility_patterns)
{
    WorkItem work_item;

    // while
}

int main(int argc, char *argv[])
{
    // Parse command-line arguments using args_parser
    ParsedArgs args;
    if (!parseArguments(argc, argv, args))
    {
        // Parsing failed; exit the program
        return EXIT_FAILURE;
    }

    // increase cuad stack size
    // cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024);

    ReadFileResult fileResult = read_file(args.filename, args.separator, args.utility);

    // Access the parsed data
    auto &filteredTransactions = fileResult.filteredTransactions;
    auto &primary = fileResult.primary;
    auto &rename = fileResult.rename;
    int max_item = fileResult.max_item;

    // Flatten filteredTransactions
    std::vector<Item> items;
    std::vector<int> start;
    std::vector<int> end;

    for (const auto &[key, val] : filteredTransactions)
    {
        start.push_back(items.size());
        for (int i = 0; i < key.size(); i++)
        {
            std::cout << key[i] << ":" << val[i] << " ";
            items.push_back({key[i], val[i]});
        }
        std::cout << "\n";
        end.push_back(items.size());
    }

    // copy items to device
    Item *d_items;
    int *d_start;
    int *d_end;
    int *d_primary;

    size_t num_items = items.size();
    size_t num_transactions = start.size();

    cudaMalloc(&d_items, num_items * sizeof(Item));
    cudaMalloc(&d_start, num_transactions * sizeof(int));
    cudaMalloc(&d_end, num_transactions * sizeof(int));
    cudaMalloc(&d_primary, primary.size() * sizeof(int));

    cudaMemcpy(d_items, items.data(), num_items * sizeof(Item), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start.data(), num_transactions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end.data(), num_transactions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(int), cudaMemcpyHostToDevice);

    int32_t *d_high_utility_patterns;
    cudaMallocManaged(&d_high_utility_patterns, 128 * MEGA); // 1GB
    d_high_utility_patterns[1] = 2;

    // memory
    // std::cout << "Allocating Memory: " << page_count * page_size << " bytes\t(MB: " << (page_count * page_size) / (MEGA) << ")\n";

    // CudaMemoryManager *memory_manager = createMemoryManager(total_memory, page_size);
    CudaMemoryManager *mm = createCudaMemoryManager(total_memory / page_size, page_size);

    // start work queue
    AtomicWorkStack *work_queue;
    cudaMalloc(&work_queue, sizeof(AtomicWorkStack));

    copy<<<1, 1>>>(mm, work_queue, d_items, d_start, d_end, d_primary, primary.size(), num_transactions, max_item);
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
        std::cout << "\n";
        std::cout << "\n";


    verify<<<1, 1>>>(mm, work_queue, args.utility);

    // // free the memory
    // cudaFree(d_items);
    // cudaFree(d_start);
    // cudaFree(d_end);
    // cudaFree(d_primary);

    // mine<<<blocks, 1>>>(memory_manager, work_queue, args.utility, d_high_utility_patterns);
    cudaDeviceSynchronize();

    std::cout << "High Utility Patterns: " << d_high_utility_patterns[0] << "\n";

    std::map<std::string, int> Patterns;

    // convert high utility patterns to string
    std::string high_utility_patterns_str = "";
    std::vector<int> high_utility_patten;
    for (int i = 0; i < d_high_utility_patterns[1]; i++)
    {
        while (d_high_utility_patterns[i + 2] != 0)
        {
            high_utility_patten.push_back(d_high_utility_patterns[i + 2]);
            i++;
        }
        // if empty, skip
        if (high_utility_patten.size() == 0)
        {
            continue;
        }

        for (int j = 0; j < high_utility_patten.size() - 1; j++)
        {
            high_utility_patterns_str += rename[high_utility_patten[j]] + " ";
        }
        Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];

        high_utility_patterns_str = "";
        high_utility_patten.clear();
    }
    for (const auto &p : Patterns)
    {
        std::cout << p.first << "UTIL: " << p.second << std::endl;
    }

    std::cout << "High Utility Patterns: " << d_high_utility_patterns[0] << "\n";

    return 0;
}