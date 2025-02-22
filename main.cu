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
#define total_memory (6 * GIGA)

// make && ./cuEFIM '/home/tarun/testing/test.txt' 5 \\s
// make && time ./cuEFIM '/home/tarun/cuEFIM/datasets/accidents_utility_spmf.txt' 15000000 \\s
__global__ void copy_work(AtomicWorkStack *curr_work_queue, WorkItem *work_item, int *primary, int primary_size)
{
    int tid = threadIdx.x;
    __shared__ WorkItem item;
    // WorkItem item;
    if (tid == 0)
    {
        item.pattern = (int *)global_malloc(sizeof(int));
        item.pattern_length = 0;
        item.utility = 0;
        item.db = (Database *)global_malloc(sizeof(Database));
        item.db->numItems = 0;
        item.db->d_data = (Item *)global_malloc(sizeof(Item) * work_item->db->numItems);
        item.db->d_transactions = (Transaction *)global_malloc(sizeof(Transaction) * work_item->db->numTransactions);
        item.db->numTransactions = work_item->db->numTransactions;
        item.max_item = work_item->max_item;

        item.work_done = (int *)global_malloc(sizeof(int));
        item.work_done[0] = 0;
        item.work_count = primary_size;
    }

    __syncthreads();

    for (int i = tid; i < work_item->db->numItems; i += blockDim.x)
    {
        item.db->d_data[i] = work_item->db->d_data[i];
    }

    __syncthreads();

    for (int i = tid; i < work_item->db->numTransactions; i += blockDim.x)
    {
        item.db->d_transactions[i].data = item.db->d_data + (work_item->db->d_transactions[i].data - work_item->db->d_data);
        item.db->d_transactions[i].utility = 0;
        item.db->d_transactions[i].length = work_item->db->d_transactions[i].length;
    }

    __syncthreads();

    if (tid == 0)
    {
        for (int i = 0; i < primary_size; i++)
        {
            item.primary = primary[i];
            curr_work_queue->push(item);
        }
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

        Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];

        high_utility_patterns_str = "";
        high_utility_patten.clear();
    }
    return Patterns;
}

int main(int argc, char *argv[])
{
    cudaError_t cudaErr;

    cudaDeviceProp deviceProp;
    cudaErr = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(cudaErr) << std::endl;
        return 1;
    }

    // Calculate the theoretical max concurrent threads
    int maxConcurrentThreads = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;

    std::cout << "Device " << 0 << ": " << deviceProp.name << std::endl;
    std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Max threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Theoretical max concurrent threads: "
              << maxConcurrentThreads << std::endl;

    int block_count = maxConcurrentThreads / deviceProp.maxThreadsPerBlock;
    std::cout << "Block count: " << maxConcurrentThreads / threads << std::endl;

    // increase cuda stack size
    // cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024);
    // Make CPU not poll
    cudaErr = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (cudaErr != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(cudaErr));
        return -1;
    }

    // increase stack size
    // cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
    Timer timer;

    // Parse command-line arguments using args_parser
    ParsedArgs args;
    if (!parseArguments(argc, argv, args))
    {
        // Parsing failed; exit the program
        return EXIT_FAILURE;
    }

    timer.recordPoint("Start");
    ReadFileResult fileResult = read_file(args.filename, args.separator, args.utility);
    timer.recordPoint("File Read");

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
            // std::cout << key[i] << ":" << val[i] << " ";
        }
        end.push_back(items.size());
        // std::cout << "\n";
    }

    int32_t *d_high_utility_patterns;
    cudaMallocManaged(&d_high_utility_patterns, 128 * MEGA);
    memset(d_high_utility_patterns, 0, 128 * MEGA);
    d_high_utility_patterns[1] = 2;

    init_global_allocator(total_memory, 0);

    std::cout << "Number of Transactions: " << start.size() << "\n";
    // CudaMemoryManager *mm = createCudaMemoryManager(total_memory, page_size);
    // std::cout << "Memory Manager Initialized\n";

    AtomicWorkStack *stack;
    cudaMallocManaged(&stack, sizeof(AtomicWorkStack));
    stack->init();

    WorkItem *work_item;
    cudaMallocManaged(&work_item, sizeof(WorkItem));
    work_item->pattern = nullptr;
    work_item->pattern_length = 0;
    // work_item->work_count = primary.size();
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
    work_item->max_item = max_item;

    tempWork *working;
    cudaMallocManaged(&working, sizeof(tempWork));
    // cudaMallocManaged(&working->db, sizeof(Database));

    cudaMalloc(&working->temp_transaction, sizeof(Transaction) * db->numTransactions);
    cudaMalloc(&working->local_util, sizeof(Item) * max_item * scale);
    cudaMalloc(&working->hashes, sizeof(int) * db->numTransactions * scale);
    cudaMalloc(&working->subtree_util, sizeof(Item) * max_item * scale);

    int *d_primary;
    cudaMalloc(&d_primary, primary.size() * sizeof(int));
    cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(int), cudaMemcpyHostToDevice);


    // int i = 0;
    // for(auto &item : primary
    for (int i = 0; i < primary.size(); i++)
    {
        int item = primary[i];

        work_item->primary = item;
        // printf("Scanning\n");

        working->num_transactions = 0;
        working->num_items = 0;
        working->utility = 0;

        cudaMemset(working->local_util, 0, sizeof(Item) * max_item * scale);
        cudaMemset(working->temp_transaction, 0, sizeof(Transaction) * start.size());
        cudaMemset(working->hashes, -1, sizeof(int) * db->numTransactions * scale);
        cudaMemset(working->subtree_util, 0, sizeof(Item) * max_item * scale);
    

        scan<<<((work_item->db->numTransactions + threads) / threads),threads>>>(work_item, working);
        cudaDeviceSynchronize();

        printf("%d:Item: %d\tUtility: %d\n", i, item, working->utility);
        if (working->utility >= args.utility)
        {
            d_high_utility_patterns[0] += 1;
            int index = d_high_utility_patterns[1];
            d_high_utility_patterns[1] += 3;
            d_high_utility_patterns[index] = item;
            d_high_utility_patterns[index + 1] = working->utility;
        }

        if (working->num_transactions == 0) continue;

    //     // printf("Copying\n");

        allocate<<<1,1>>>(working);
        cudaDeviceSynchronize();

        trim_project<<<((working->num_transactions + threads) / threads),threads>>>(work_item, working, args.utility);
        cudaDeviceSynchronize();

    //     // printf("Project Trim Done\n");
        finalize<<<1,32>>>(stack, work_item, working, args.utility);
        cudaDeviceSynchronize();
    //     // printf("\n\n");
    //     i++;
    }

    while (stack->active > 0)
    {
        printf("Top: %d\n", stack->active);
        mine<<<blocks, threads>>>(stack, d_high_utility_patterns, args.utility);
        // print last error
        cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess)
        {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaErr));
            return -1;
        }

        cudaDeviceSynchronize();
    }

    timer.recordPoint("Kernel Execution");
    std::map<std::string, int> Patterns = parse_patterns(d_high_utility_patterns, rename);
    // timer
    cudaFree(d_high_utility_patterns);

    // for (const auto &p : Patterns)
    // {
    //     std::cout << p.first << "UTIL: " << p.second << std::endl;
    // }

    std::cout << "High Utility Patterns: " << Patterns.size() << "\n";

    // print_global_stats();

    free_global_allocator();

    timer.printRecords();

    return 0;
}