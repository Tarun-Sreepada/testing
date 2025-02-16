
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

// #include "device/Ouroboros_impl.cuh"
// #include "device/MemoryInitialization.cuh"
// #include "InstanceDefinitions.cuh"
// #include "Utility.h"
/*

b e c UTIL: 45
d b UTIL: 44
d b c UTIL: 51
d b e UTIL: 53
d b e c UTIL: 60
d e c UTIL: 45

*/

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define page_size 512
#define total_memory 24 * GIGA

#define blocks 512
#define threads 16
//  1959  make && ./cuEFIM '/home/tarun/testing/test.txt' 5 \\s
//  1960  make && time ./cuEFIM '/home/tarun/cuEFIM/datasets/accidents_utility_spmf.txt' 15000000 \\s

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
    std::cout << "Duplicate: " << duplicate << "\n";
    std::cout << "Util Duplicate: " << util_dup << "\n";
    return Patterns;
}

// copy<<<1,1>>>(d_items, d_start, d_end, d_primary, items.size(), start.size(), primary.size(), max_item, d_high_utility_patterns, mm, curr_work_queue, args.utility);
// template <typename MemoryManagerType>
__global__ void copy(Item *d_items, int *d_start, int *d_end, int *d_primary, int items_size, int start_size, int primary_size, int max_item, int *d_high_utility_patterns, CudaMemoryManager *mm, AtomicWorkStack *curr_work_queue, int utility)
{

    WorkItem work_item;
    work_item.pattern = (int *)mm->malloc(sizeof(int));
    work_item.pattern_length = 0;

    work_item.db = (Database *)mm->malloc(sizeof(Database));
    work_item.db->numItems = items_size;

    work_item.db->d_data = (Item *)mm->malloc(items_size * sizeof(Item));
    memcpy(work_item.db->d_data, d_items, items_size * sizeof(Item));

    // work_item.db->d_transactions = reinterpret_cast<Transaction *>(mm.hostMalloc(start_size * sizeof(Transaction)));
    work_item.db->d_transactions = (Transaction *)mm->malloc(start_size * sizeof(Transaction));
    work_item.db->numTransactions = start_size;
    for (int i = 0; i < start_size; i++)
    {
        work_item.db->d_transactions[i].data = work_item.db->d_data + d_start[i];
        work_item.db->d_transactions[i].utility = 0;
        work_item.db->d_transactions[i].length = d_end[i] - d_start[i];
    }

    work_item.db->numItems = items_size;

    // work_item.work_done = reinterpret_cast<int *>(mm.hostMalloc(sizeof(int)));
    work_item.work_done = (int *)mm->malloc(sizeof(int));
    work_item.work_count = primary_size;
    work_item.max_item = max_item;

    for (int i = 0; i < primary_size; i++)
    {
        work_item.primary = d_primary[i];
        // curr_work_queue[i]->push(work_item);
        curr_work_queue->push(work_item);
    }


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
    cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024);

    // Parse command-line arguments using args_parser
    ParsedArgs args;
    if (!parseArguments(argc, argv, args))
    {
        // Parsing failed; exit the program
        return EXIT_FAILURE;
    }

    // increase cuda stack size
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
            // std::cout << key[i] << ":" << val[i] << " ";
            items.push_back({key[i], val[i]});
        }
        // std::cout << "\n";
        end.push_back(items.size());
    }
    // std::cout << "\n";

    int32_t *d_high_utility_patterns;
    cudaMallocManaged(&d_high_utility_patterns, 128 * MEGA);
    memset(d_high_utility_patterns, 0, 128 * MEGA);
    d_high_utility_patterns[1] = 2;

    // using MemoryManagerType = OuroVAPQ;
    // MemoryManagerType mm;
    // mm.initialize(0, 1024 * 1024 * 1024);

    CudaMemoryManager *mm = createCudaMemoryManager(total_memory / page_size, page_size);

    Item *d_items;
    int *d_start;
    int *d_end;
    int *d_primary;

    cudaMalloc(&d_items, items.size() * sizeof(Item));
    cudaMalloc(&d_start, start.size() * sizeof(int));
    cudaMalloc(&d_end, end.size() * sizeof(int));
    cudaMalloc(&d_primary, primary.size() * sizeof(int));

    // copy items to device
    cudaMemcpy(d_items, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start.data(), start.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end.data(), end.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(int), cudaMemcpyHostToDevice);

    // AtomicWorkStack *curr_work_queue;
    // cudaMallocManaged(&curr_work_queue, sizeof(AtomicWorkStack));
    AtomicWorkStack *curr_work_queue;
    cudaMallocManaged(&curr_work_queue, sizeof(AtomicWorkStack));
    curr_work_queue->init();
    // allocate as many work queues as there are primary items
    // cudaMallocManaged(&curr_work_queue, primary.size() * sizeof(AtomicWorkStack));
    // for (int i = 0; i < primary.size(); i++)
    // {
    //     cudaMallocManaged(&curr_work_queue[i], sizeof(AtomicWorkStack));
    //     curr_work_queue[i]->init();
    // }


    copy<<<1,1>>>(d_items, d_start, d_end, d_primary, items.size(), start.size(), primary.size(), max_item, d_high_utility_patterns, mm, curr_work_queue, args.utility);
    cudaDeviceSynchronize();

    cudaFree(d_items);
    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_primary);


    // cudaError_t cudaStatus = cudaGetLastError();
    printf("Top: %d\n", curr_work_queue->active);
    auto starttime = std::chrono::high_resolution_clock::now();

    test<<<blocks, threads>>>(curr_work_queue, d_high_utility_patterns, mm, args.utility);
    cudaDeviceSynchronize();
    printf("Top: %d\n", curr_work_queue->active);

    auto endtime = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0 << " s\n";


    // while (curr_work_queue->active > 0)
    {
        // curr_work_queue->top
        
        //     cudaStatus = cudaGetLastError();
        //     if (cudaStatus != cudaSuccess)
        //     {
        //         fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //         return -1;

        // std::cout << "Next Work Queue Size: " << new_work_queue->top << "\n";

        //     // clear work queue make it ready for next iteration
        // memcpy(curr_work_queue, new_work_queue, sizeof(AtomicWorkStack));
        // memset(new_work_queue, 0, sizeof(AtomicWorkStack));
    }

    std::cout << "High Utility Patterns: " << d_high_utility_patterns[0] << "\n";
    std::map<std::string, int> Patterns = parse_patterns(d_high_utility_patterns, rename);
    cudaFree(d_high_utility_patterns);

    // for (const auto &p : Patterns)
    // {
    //     std::cout << p.first << "UTIL: " << p.second << std::endl;
    // }
    std::cout << "High Utility Patterns: " << Patterns.size() << "\n";

    return 0;
}