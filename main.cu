
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
#include "mine.cuh"

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define page_size 128
#define total_memory 2 * GIGA

#define blocks 2




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
            // std::cout << key[i] << ":" << val[i] << " ";
            items.push_back({key[i], val[i]});
        }
        // std::cout << "\n";
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

    mine<<<blocks, 1>>>(mm, work_queue, args.utility, d_high_utility_patterns);

    // // free the memory
    // cudaFree(d_items);
    // cudaFree(d_start);
    // cudaFree(d_end);
    // cudaFree(d_primary);

    cudaDeviceSynchronize();
    cudaFree(d_items);
    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_primary);

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