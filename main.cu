
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
// include iota
#include <numeric>

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

// #define page_size (128 * KILO)
#define total_memory (32 * GIGA)

#include <limits>
#include <iostream>

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
    cudaError_t cudaStatus;

    int gpu_max_shared_mem;
    cudaDeviceGetAttribute(&gpu_max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    std::cout << "Max Shared Memory: " << gpu_max_shared_mem << " Bytes\n";



    // increase cuda stack size
    // cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024);
    // Make CPU not poll
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(err));
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

    int max_size = 0;
    for (const auto &[key, val] : filteredTransactions)
    {
        max_size = std::max(max_size, (int)key.size());
        start.push_back(items.size());
        for (int i = 0; i < key.size(); i++)
        {
            items.push_back({key[i], val[i]});
        }
        end.push_back(items.size());
    }
    std::cout << "Largest Transaction: " << max_size << "\n";

    std::vector<int> transactionIndices(start.size());
    std::iota(transactionIndices.begin(), transactionIndices.end(), 0);

    std::sort(transactionIndices.begin(), transactionIndices.end(),
              [&](int i, int j)
              {
                  return (end[i] - start[i]) > (end[j] - start[j]);
              });

    // Rebuild the items, start, and end vectors in the new order.
    std::vector<Item> sortedItems;
    std::vector<int> sortedStart, sortedEnd;

    for (int idx : transactionIndices)
    {
        sortedStart.push_back(sortedItems.size());
        for (int i = start[idx]; i < end[idx]; ++i)
        {
            sortedItems.push_back(items[i]);
        }
        sortedEnd.push_back(sortedItems.size());
    }

    // Replace the original vectors with the sorted ones.
    items = std::move(sortedItems);
    start = std::move(sortedStart);
    end = std::move(sortedEnd);

    // Optionally, print out the size of the largest transaction.
    if (!start.empty())
    {
        int largestTransactionSize = end[0] - start[0];
        std::cout << "Largest Transaction: " << largestTransactionSize << "\n";
    }

    /*
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
    */

    int shared_mem_req = max_item * sizeof(Utils) * scale // for local_util
                        + max_size * sizeof(Item) // for temp_transaction
                         + 3 * KILO; // for other variables

    std::cout << "Shared Memory Required: " << shared_mem_req << " Bytes\n";

    if (shared_mem_req > gpu_max_shared_mem)
    {
        std::cout << "Requested shared memory exceeds GPU max;\n";
        // shared_mem_req = gpu_max_shared_mem;
        return -1;
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

    copy_work<<<1, args.threads>>>(curr_work_queue, work_item, d_primary, primary.size());
    cudaDeviceSynchronize();
    timer.recordPoint("Data Copy to GPU");

    cudaFree(d_primary);
    cudaFree(db->d_data);
    cudaFree(db->d_transactions);
    cudaFree(db);

    time_stuff *d_time_stuff;
    cudaMallocManaged(&d_time_stuff, sizeof(time_stuff) * args.blocks);
    memset(d_time_stuff, 0, sizeof(time_stuff) * args.blocks);

    cudaMemAdvise(d_time_stuff, sizeof(time_stuff) * args.blocks, cudaMemAdviseSetPreferredLocation, 0); // set preferred location to GPU

    core_flame_graph *core_graphs;
    cudaMallocManaged(&core_graphs, sizeof(core_flame_graph) * args.blocks);
    memset(core_graphs, 0, sizeof(core_flame_graph) * args.blocks);

    // while (curr_work_queue->active > 0)
    // {
        printf("Top: %d\n", curr_work_queue->active);
        test<<<args.blocks, args.threads, shared_mem_req>>>(curr_work_queue, d_high_utility_patterns, args.utility, d_time_stuff, core_graphs);
        // print last error
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return -1;
        }

        cudaDeviceSynchronize();
    // }

    timer.recordPoint("Kernel Execution");
    std::map<std::string, int> Patterns = parse_patterns(d_high_utility_patterns, rename);
    // timer
    cudaFree(d_high_utility_patterns);

    // for (const auto &p : Patterns)
    // {
    //     std::cout << p.first << "UTIL: " << p.second << std::endl;
    // }

    // get GPU clock

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU Name: " << prop.name << "\n";
    uint64_t clock_rate = prop.clockRate * 1000;
    std::cout << "GPU Clock Rate: " << clock_rate / 1000 << " KHz\n";

    std::cout << "\n";


    
    // Assume d_time_stuff is defined and populated, and clock_rate and blocks are defined.
    
    float total_scan_time = 0.0f;
    float total_merge_time = 0.0f;
    float total_memory_time = 0.0f;
    float total_idle_time = 0.0f;
    
    // Initialize min values to a very high value and max values to 0.
    float min_idle_time = std::numeric_limits<float>::max();
    float max_idle_time = 0.0f;
    float min_memory_time = std::numeric_limits<float>::max();
    float max_memory_time = 0.0f;
    float min_scan_time = std::numeric_limits<float>::max();
    float max_scan_time = 0.0f;
    float min_merge_time = std::numeric_limits<float>::max();
    float max_merge_time = 0.0f;
    int min_processed_count = std::numeric_limits<int>::max();
    int max_processed_count = 0;
    float min_push_time = std::numeric_limits<float>::max();
    float max_push_time = 0.0f;


    for (int i = 0; i <args.blocks; i++)
    {
        // Calculate elapsed time for the block (if needed).
        float elapsed_time = (float)d_time_stuff[i].end - (float)d_time_stuff[i].start;
        elapsed_time /= (float)clock_rate;
        
        // Calculate each specific time metric.
        float idle_time   = d_time_stuff[i].idle        / (float)clock_rate;
        float memory_time = d_time_stuff[i].memory_alloc/ (float)clock_rate;
        float scan_time   = d_time_stuff[i].scanning      / (float)clock_rate;
        float merge_time  = d_time_stuff[i].merging       / (float)clock_rate;
        float longest_scan = d_time_stuff[i].tt_scan / (float)clock_rate;
        float longest_merge = d_time_stuff[i].tt_merging / (float)clock_rate;
        float push_time = d_time_stuff[i].push / (float)clock_rate;
    
        // Print per-block times.
        std::cout << "Block: " << i << " Time: " << elapsed_time << " s\n";
        std::cout << "Block: " << i << " Processed: " << d_time_stuff[i].processed_count << "\n";
        std::cout << "Block: " << i << " Idle: " << idle_time << " s\n";
        std::cout << "Block: " << i << " Memory Alloc: " << memory_time << " s\n";
        std::cout << "Block: " << i << " Scan: " << scan_time << " s\n";
        std::cout << "Block: " << i << " Merge: " << merge_time << " s\n";
        std::cout << "Block: " << i << " Longest Scan: " << longest_scan << "\n";
        std::cout << "Block: " << i << " Longest Scan transaction size: " << d_time_stuff[i].largest_trans_scan << "\n";
        std::cout << "Block: " << i << " Longest Merge transaction count: " << d_time_stuff[i].largest_trans_count_scan << "\n";
        std::cout << "Block: " << i << " Longest Merge: " <<  longest_merge << "\n";
        std::cout << "Block: " << i << " Longest Merge transaction size: " << d_time_stuff[i].largest_trans_merge << "\n";
        std::cout << "Block: " << i << " Longest Merge transaction count: " << d_time_stuff[i].largest_trans_count_merge << "\n";
        std::cout << "Block: " << i << " Longest Merge merge count: " << d_time_stuff[i].merge_count << "\n";
        std::cout << "Block: " << i << " Push: " << push_time << "\n";

        std::cout << "===========================================================================\n";
        std::cout << "Idle | Scan | Memory | Merge | Push\n";
        for (int j = 0; j < d_time_stuff[i].processed_count + 1; j++)
        {
            float idle = core_graphs[i].buckets[j].idle / (float)clock_rate;
            float scan = core_graphs[i].buckets[j].scanning / (float)clock_rate;
            float memory = core_graphs[i].buckets[j].memory_alloc / (float)clock_rate;
            float merge = core_graphs[i].buckets[j].merging / (float)clock_rate;
            float push = core_graphs[i].buckets[j].push / (float)clock_rate;
            std::cout << idle << " | " << scan << " | " << memory << " | " << merge << " | " << push << "\n";

            // std::cout << core_graphs[i].buckets[j].idle << " | " << core_graphs[i].buckets[j].scanning << " | " << core_graphs[i].buckets[j].memory_alloc << " | " << core_graphs[i].buckets[j].merging << " | " << core_graphs[i].buckets[j].push << "\n";
        }
        std::cout << "===========================================================================\n\n";
    
        // Accumulate totals for averages.
        total_idle_time   += idle_time;
        total_memory_time += memory_time;
        total_scan_time   += scan_time;
        total_merge_time  += merge_time;
    
        // Update minimum values.
        if (idle_time < min_idle_time)   min_idle_time   = idle_time;
        if (memory_time < min_memory_time) min_memory_time = memory_time;
        if (scan_time < min_scan_time)     min_scan_time   = scan_time;
        if (merge_time < min_merge_time)   min_merge_time  = merge_time;
        if (d_time_stuff[i].processed_count < min_processed_count) min_processed_count = d_time_stuff[i].processed_count;
        if (push_time < min_push_time) min_push_time = push_time;
    
        // Update maximum values.
        if (idle_time > max_idle_time)   max_idle_time   = idle_time;
        if (memory_time > max_memory_time) max_memory_time = memory_time;
        if (scan_time > max_scan_time)     max_scan_time   = scan_time;
        if (merge_time > max_merge_time)   max_merge_time  = merge_time;
        if (d_time_stuff[i].processed_count > max_processed_count) max_processed_count = d_time_stuff[i].processed_count;
        if (push_time > max_push_time) max_push_time = push_time;
    }
    
    // Compute averages.
    float avg_idle_time   = total_idle_time /args.blocks;
    float avg_memory_time = total_memory_time /args.blocks;
    float avg_scan_time   = total_scan_time /args.blocks;
    float avg_merge_time  = total_merge_time /args.blocks;
    
    // Print summary statistics.
    std::cout << "Idle Time - Avg: "   << avg_idle_time   << " s, Min: " << min_idle_time   << " s, Max: " << max_idle_time   << " s\n";
    std::cout << "Memory Time - Avg: " << avg_memory_time << " s, Min: " << min_memory_time << " s, Max: " << max_memory_time << " s\n";
    std::cout << "Scan Time - Avg: "   << avg_scan_time   << " s, Min: " << min_scan_time   << " s, Max: " << max_scan_time   << " s\n";
    std::cout << "Merge Time - Avg: "  << avg_merge_time  << " s, Min: " << min_merge_time  << " s, Max: " << max_merge_time  << " s\n";
    std::cout << "Push Time - Min: " << min_push_time << ", Max: " << max_push_time << "\n";
    std::cout << "Processed Count - Min: " << min_processed_count << ", Max: " << max_processed_count << "\n";
    

    
    std::cout << "Blocks: " <<args.blocks << "\n";
    std::cout << "Threads per block: " << args.threads << "\n";

    std::cout << "High Utility Patterns: " << Patterns.size() << "\n";

    // print_global_stats();

    free_global_allocator();

    timer.printRecords();

    return 0;
}