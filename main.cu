
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include "args.h"     // Include the args parser
#include "parser.h"   // Include the file reader
#include "work.cuh"   // Include the work queue
#include "memory.cuh" // Include the memory manager
#include "mine.cuh"

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define page_size (128 * KILO)
#define total_memory (4 * GIGA)

#define blocks 512
#define threads 512
// make && ./cuEFIM '/home/tarun/testing/test.txt' 5 \\s
// make && time ./cuEFIM '/home/tarun/cuEFIM/datasets/accidents_utility_spmf.txt' 15000000 \\s



__global__ void copy_work(AtomicWorkStack *curr_work_queue, WorkItem *work_item, int *primary, int primary_size)
{
    WorkItem item;
    item.pattern = (int *)global_malloc(sizeof(int));
    item.pattern_length = 0;
    item.utility = 0;
    item.db = (Database *)global_malloc(sizeof(Database));
    item.db->numItems = 0;

    item.db->d_data = (Item *)global_malloc(sizeof(Item) * work_item->db->numItems);
    memcpy(item.db->d_data, work_item->db->d_data, sizeof(Item) * work_item->db->numItems);
    
    item.db->d_transactions = (Transaction *)global_malloc(sizeof(Transaction) * work_item->db->numTransactions);
    for (int i = 0; i < work_item->db->numTransactions; i++)
    {
        item.db->d_transactions[i].data = item.db->d_data + (work_item->db->d_transactions[i].data - work_item->db->d_data);
        item.db->d_transactions[i].utility = 0;
        item.db->d_transactions[i].length = work_item->db->d_transactions[i].length;
    }
    item.db->numTransactions = work_item->db->numTransactions;
    item.max_item = work_item->max_item;

    item.work_done = (int *)global_malloc(sizeof(int));
    item.work_done[0] = 0;
    item.work_count = primary_size;

    for (int i = 0; i < primary_size; i++)
    {
        item.primary = primary[i];
        curr_work_queue->push(item);
    }

}


std::map<std::string, int> parse_patterns(int *d_high_utility_patterns, std::unordered_map<int, std::string> rename)

{
    std::map<std::string, int> Patterns;
    int duplicate = 0;
    int util_dup = 0;

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

        //

        // Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];
        if (Patterns.find(high_utility_patterns_str) == Patterns.end())
        {
            Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];
        }
        else
        {
            duplicate++;
            // print old util and new util
            // std::cout << "Old: " << Patterns[high_utility_patterns_str] << " New: " << high_utility_patten[high_utility_patten.size() - 1] << "\n";
            if (Patterns[high_utility_patterns_str] == high_utility_patten[high_utility_patten.size() - 1])
            {
                util_dup++;
            }
        }

        high_utility_patterns_str = "";
        high_utility_patten.clear();
    }
    return Patterns;
}

int main(int argc, char *argv[])
{
    // Make CPU not poll
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // increase stack size
    // cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);

    // Parse command-line arguments using args_parser
    ParsedArgs args;
    if (!parseArguments(argc, argv, args))
    {
        // Parsing failed; exit the program
        return EXIT_FAILURE;
    }

    // increase cuda stack size
    cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024);

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
            items.push_back({key[i], val[i]});
        }
        end.push_back(items.size());
    }

    int32_t *d_high_utility_patterns;
    cudaMallocManaged(&d_high_utility_patterns, 128 * MEGA);
    memset(d_high_utility_patterns, 0, 128 * MEGA);
    d_high_utility_patterns[1] = 2;

    init_global_allocator(total_memory, 0);

    // CudaMemoryManager *mm = createCudaMemoryManager(total_memory, page_size);
    // std::cout << "Memory Manager Initialized\n";

    AtomicWorkStack *curr_work_queue;
    cudaMallocManaged(&curr_work_queue, sizeof(AtomicWorkStack));
    curr_work_queue->init();


    WorkItem *work_item;
    cudaMallocManaged(&work_item, sizeof(WorkItem));
    work_item->pattern = nullptr;
    work_item->pattern_length = 0;
    work_item->work_count = primary.size();
    work_item->max_item = max_item;
    work_item->work_done = nullptr;

    Database *db;
    cudaMallocManaged(&db, sizeof(Database));
    db->numItems = items.size();
    db->d_data = nullptr;
    db->d_transactions = nullptr;
    db->numTransactions = start.size();

    cudaMallocManaged(&db->d_data, items.size() * sizeof(Item));
    cudaMemcpy(db->d_data, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice);

    cudaMallocManaged(&db->d_transactions, start.size() * sizeof(Transaction));
    for (int i = 0; i < start.size(); i++)
    {
        db->d_transactions[i].data = db->d_data + start[i];
        db->d_transactions[i].utility = 0;
        db->d_transactions[i].length = end[i] - start[i];
    }

    work_item->db = db;

    int *d_primary;
    cudaMallocManaged(&d_primary, primary.size() * sizeof(int));
    cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(int), cudaMemcpyHostToDevice);
    

    copy_work<<<1, 1>>>(curr_work_queue, work_item, d_primary, primary.size());
    cudaDeviceSynchronize();

    // WorkItem work_item;
    // work_item.pattern = (int *)mm->host_malloc(sizeof(int));
    // work_item.pattern_length = 0;


    // work_item.db = (Database *)mm->host_malloc(sizeof(Database));
    // work_item.db->numItems = items.size();


    // work_item.db->d_data = (Item *)mm->host_malloc(items.size() * sizeof(Item));
    // cudaMemcpy(work_item.db->d_data, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice);

    // // work_item.db->d_transactions = reinterpret_cast<Transaction *>(mm.hostMalloc(start.size() * sizeof(Transaction)));
    // work_item.db->d_transactions = (Transaction *)mm->host_malloc(start.size() * sizeof(Transaction));
    // work_item.db->numTransactions = start.size();
    // for (int i = 0; i < start.size(); i++)
    // {
    //     work_item.db->d_transactions[i].data = work_item.db->d_data + start[i];
    //     work_item.db->d_transactions[i].utility = 0;
    //     work_item.db->d_transactions[i].length = end[i] - start[i];
    // }

    // work_item.db->numItems = items.size();

    // // work_item.work_done = reinterpret_cast<int *>(mm.hostMalloc(sizeof(int)));
    // work_item.work_done = (int *)mm->host_malloc(sizeof(int));
    // work_item.work_count = primary.size();
    // work_item.max_item = max_item;

    // cudaDeviceSynchronize();

    // for (int i = 0; i < primary.size(); i++)
    // {
    //     work_item.primary = primary[i];
    //     curr_work_queue->host_push(work_item);
    // }

    auto starttime = std::chrono::high_resolution_clock::now();
    cudaError_t cudaStatus;

    while (curr_work_queue->active > 0)
    {
        printf("Top: %d\n", curr_work_queue->active);
        test<<<blocks, threads>>>(curr_work_queue, d_high_utility_patterns, args.utility);
        // print last error
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return -1;
        }

        cudaDeviceSynchronize();
    }

    auto endtime = std::chrono::high_resolution_clock::now();

    std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0 << " s\n";

    // std::cout << "High Utility Patterns: " << d_high_utility_patterns[0] << "\n";
    std::map<std::string, int> Patterns = parse_patterns(d_high_utility_patterns, rename);
    cudaFree(d_high_utility_patterns);

    // for (const auto &p : Patterns)
    // {
    //     std::cout << p.first << "UTIL: " << p.second << std::endl;
    // }

    std::cout << "High Utility Patterns: " << Patterns.size() << "\n";


    free_global_allocator();

    return 0;
}