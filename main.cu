
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "args.h"  // Include the args parser
#include "parser.h"  // Include the file reader
#include "work.cuh"  // Include the work queue


#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "Utility.h"

const int work_queue_size = 16 * 1024; // 16K, can be adjusted
const int scale_factor = 2;

__device__ uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ uint32_t hashFunction(uint32_t key, uint32_t tableSize)
{
    return pcg_hash(key) % tableSize;
}

template <typename MemoryManagerType>
__global__ void initial_copy( 
    MemoryManagerType* memory_manager, 
    AtomicWorkQueue<WorkItem, 16 * 1024>* work_queue,
    Item* items,
    int* start,
    int* end,
    int* primary,
    int num_primary,
    int num_transactions,
    int max_item)
{

    // int *pattern = memory_manager->alloc(1);
    // int *pattern = memory_manager->malloc(1);
    int *pattern = reinterpret_cast<int*>(memory_manager->malloc(1));
    pattern[0] = 0;
    // int *work_done = memory_manager->malloc(1);
    int *work_done = reinterpret_cast<int*>(memory_manager->malloc(1));
    int *utility = reinterpret_cast<int*>(memory_manager->malloc(num_transactions * sizeof(int)));
    work_done[0] = 0;


    int *n_start = reinterpret_cast<int*>(memory_manager->malloc(num_transactions * sizeof(int)));
    int *n_end = reinterpret_cast<int*>(memory_manager->malloc(num_transactions * sizeof(int)));
    // int *n_items = reinterpret_cast<int*>(memory_manager->malloc((end[num_transactions] - start[0]) * sizeof(Item)));
    Item *n_items = reinterpret_cast<Item*>(memory_manager->malloc((end[num_transactions - 1] - start[0]) * sizeof(Item)));


    // copy items
    // for (int i = 0; i < num_transactions; i++) {
    //     n_start[i] = start[i];
    //     n_end[i] = end[i];
    //     for (int j = start[i]; j < end[i]; j++) {
    //         n_items[j] = items[j];
    //     }
    // }
    memcpy(n_start, start, num_transactions * sizeof(int));
    memcpy(n_end, end, num_transactions * sizeof(int));
    memcpy(n_items, items, (end[num_transactions - 1] - start[0]) * sizeof(Item));




    for (int i = 0; i < num_primary; i++) {
        WorkItem work_item;
        work_item.pattern = pattern;
        // work_item.items = items;
        work_item.items = n_items;
        work_item.num_items = n_end[num_transactions-1] - n_start[0];
        // printf("Num Items: %d\tEnd: %d\tStart: %d\n", work_item.num_items, n_end[num_transactions], n_start[0]);
        // work_item.start = start;
        work_item.start = n_start;
        // work_item.end = end;
        work_item.end = n_end;
        work_item.utility = utility;
        memset(work_item.utility, 0, num_transactions * sizeof(uint32_t));
        work_item.num_transactions = num_transactions;
        work_item.primary_item = primary[i];
        work_item.max_item = max_item;
        work_item.work_done = work_done;
        work_item.work_count = num_primary;



        work_queue->enqueue(work_item);
    }
}

__device__ int binarySearchItems(const Item *items, int n, uint32_t search_id, int offset, int length)
{
    // Validate that the provided range is within bounds.
    if (offset < 0 || offset >= n || length <= 0 || (offset + length) > n)
    {
        // printf("Invalid range: offset=%d, length=%d, n=%d\n", offset, length, n);
        return -1;
    }

    int l = offset;
    int r = offset + length - 1;
    while (l <= r)
    {
        int mid = l + (r - l) / 2;
        uint32_t mid_id = items[mid].key;
        if (mid_id == search_id)
        {
            return mid;
        }
        else if (mid_id < search_id)
        {
            l = mid + 1;
        }
        else
        {
            r = mid - 1;
        }
    }
    return -1;
}



template <typename MemoryManagerType>
__global__ void initial_test(MemoryManagerType* memory_manager, AtomicWorkQueue<WorkItem, 16 * 1024>* work_queue, int min_util, int *high_utility_patterns)
{
    WorkItem work_item;

    while (work_queue->get_work_count()){
        if (work_queue->dequeue(work_item)) {
            // printf("DB: \n");
            // for (int i = 0; i < work_item.num_transactions; i++) {
            //     for (int j = work_item.start[i]; j < work_item.end[i]; j++) {
            //         printf("%d:%d ", work_item.items[j].key, work_item.items[j].util);
            //     }
            //     printf("\n");
            // }
            // printf("\n");


            int *n_pattern = reinterpret_cast<int*>(memory_manager->malloc((work_item.pattern[0] + 1) * sizeof(int)));
            memcpy(n_pattern, work_item.pattern, (work_item.pattern[0] + 1) * sizeof(int));
            n_pattern[0] += 1;
            n_pattern[n_pattern[0]] = work_item.primary_item;
            printf("Pattern: ");
            for (int i = 0; i < n_pattern[0]; i++) {
                printf("%d ", n_pattern[i + 1]);
            }
            printf("\n");


            Item *n_items = reinterpret_cast<Item*>(memory_manager->malloc(work_item.num_items * sizeof(Item)));
            int *n_start = reinterpret_cast<int*>(memory_manager->malloc(work_item.num_transactions * sizeof(int)));
            int *n_end = reinterpret_cast<int*>(memory_manager->malloc(work_item.num_transactions * sizeof(int)));
            int *n_utility = reinterpret_cast<int*>(memory_manager->malloc(work_item.num_transactions * sizeof(int)));
            Item *local_util = reinterpret_cast<Item*>(memory_manager->malloc(work_item.max_item * sizeof(Item) * scale_factor));
            memset(local_util, 0, work_item.max_item * sizeof(Item) * scale_factor);
            Item *subtree_util = reinterpret_cast<Item*>(memory_manager->malloc(work_item.max_item * sizeof(Item) * scale_factor));
            memset(subtree_util, 0, work_item.max_item * sizeof(Item) * scale_factor);
            int *work_done = reinterpret_cast<int*>(memory_manager->malloc(1));


            int item_counter = 0;
            int transaction_counter = 0;
            int pattern_utility = 0;



            // printf("Projection: \n");
            for (int i = 0; i < work_item.num_transactions; i++) {
                if (work_item.start[i] == work_item.end[i]) continue;
                int idx = binarySearchItems(work_item.items, work_item.num_items, work_item.primary_item, work_item.start[i], work_item.end[i] - work_item.start[i]);
                // printf("Num Items: %d\\tPrimary Item: %d\tStart: %d\tEnd: %d\n", work_item.num_items, work_item.primary_item, work_item.start[i], work_item.end[i]);
                // printf("Idx: %d\n", idx);
                if (idx == -1) continue;

                // for (int j = idx; j < work_item.end[i]; j++) {
                //     printf("%d:%d ", work_item.items[j].key, work_item.items[j].util);
                // }
                // printf("\n");

                n_start[transaction_counter] = item_counter;
                n_utility[transaction_counter] = work_item.utility[i] + work_item.items[idx].util;
                pattern_utility += n_utility[transaction_counter];
                int temp_util = n_utility[transaction_counter];

                for (int j = idx+1; j < work_item.end[i]; j++) {
                    n_items[item_counter] = work_item.items[j];
                    temp_util += work_item.items[j].util;
                    item_counter++;
                }
                n_end[transaction_counter] = item_counter;
                transaction_counter++;

                for (int j = idx+1; j < work_item.end[i]; j++) {
                    uint32_t hash = hashFunction(work_item.items[j].key, work_item.max_item);
                    while (true)
                    {
                        if(local_util[hash].key == 0 || local_util[hash].key == work_item.items[j].key)
                        {
                            local_util[hash].key = work_item.items[j].key;
                            local_util[hash].util += temp_util;
                            break;
                        }
                        hash = (hash + 1) % (work_item.max_item * scale_factor);
                    }
                }

                // if (item)


            }
            


            printf("Pattern Utility: %d\n", pattern_utility);
            if (pattern_utility >= min_util)
            {
                int ret = atomicAdd(&high_utility_patterns[0], 1);
                printf("High Utility Patterns: %d\n", high_utility_patterns[0]);
                int offset = atomicAdd(&high_utility_patterns[1], n_pattern[0] + 2);
                // // print pattern
                // for (int i = 0; i < n_pattern[0]; i++) {
                //     printf("%d ", n_pattern[i + 1]);
                // }
                // printf("UTIL: %d\n", pattern_utility);

                for (int i = 0; i < n_pattern[0]; i++) {
                    high_utility_patterns[offset + i] = n_pattern[i + 1];
                }
                high_utility_patterns[offset + n_pattern[0]] = pattern_utility;

            }
            // // printf local utility
            // printf("Local Utility: \n");
            // for (int i = 0; i < work_item.max_item * scale_factor; i++) {
            //     if (local_util[i].key != 0) {
            //         printf("%d:%d ", local_util[i].key, local_util[i].util);
            //     }
            // }
            // printf("\n");


            if (!transaction_counter) 
            {
                int ret = atomicAdd(work_item.work_done, 1);
                if (ret == work_item.work_count - 1) {
                    memory_manager->free(work_item.pattern);
                    memory_manager->free(work_item.work_done);
                    memory_manager->free(work_item.utility);
                    memory_manager->free(work_item.start);
                    memory_manager->free(work_item.end);
                    memory_manager->free(work_item.items);
                }

                atomicSub(&work_queue->work_count, 1);
                continue;

            }



            int local_util_counter = 0;
            for (int i = 0; i < work_item.max_item * scale_factor; i++) {
                if (local_util[i].util >= min_util) {
                    local_util_counter++;
                }
            }
            // printf("Local Utility Counter: %d\n", local_util_counter);


            // allocate hash table/array
            // Item *ha


            int new_item_counter = 0;
            int new_transaction_counter = 0;

            for (int i = 0; i < work_item.num_transactions; i++) {
                int start = new_item_counter;
                for (int j = n_start[i]; j < n_end[i]; j++) {
                    uint32_t hash = hashFunction(n_items[j].key, work_item.max_item * scale_factor);

                    while(true)
                    {
                        if (local_util[hash].key == n_items[j].key)
                        {
                            if (local_util[hash].util >= min_util)
                            {
                                n_items[new_item_counter] = n_items[j];
                                new_item_counter++;
                            }
                            break;
                        }
                        hash = (hash + 1) % (work_item.max_item * scale_factor);
                    }
                    
                }

                n_start[new_transaction_counter] = start;
                n_end[new_transaction_counter] = new_item_counter;
                n_utility[new_transaction_counter] = n_utility[i];
                new_transaction_counter++;
                if (new_transaction_counter == transaction_counter) break;

            }

            // // print trimmed projection
            // printf("Trimmed Projection: \n");
            // for (int i = 0; i < new_transaction_counter; i++) {
            //     for (int j = n_start[i]; j < n_end[i]; j++) {
            //         printf("%d:%d ", n_items[j].key, n_items[j].util);
            //     }
            //     printf("\n");
            // }

            // start subtree utility
            for (int i = 0; i < new_transaction_counter; i++) {

                int temp_util = n_utility[i];
                for (int j = n_start[i]; j < n_end[i]; j++) {
                    temp_util += n_items[j].util;
                }

                int temp = 0;

                for (int j = n_start[i]; j < n_end[i]; j++) {
                    uint32_t hash = hashFunction(n_items[j].key, work_item.max_item * scale_factor);
                    while (true)
                    {
                        if(subtree_util[hash].key == 0 || subtree_util[hash].key == n_items[j].key)
                        {
                            subtree_util[hash].key = n_items[j].key;
                            subtree_util[hash].util += temp_util - temp;
                            temp = n_items[j].util;
                            break;
                        }
                        hash = (hash + 1) % (work_item.max_item * scale_factor);
                    }
                }
            }

            // // print subtree utility
            // printf("Subtree Utility: \n");
            // for (int i = 0; i < work_item.max_item * scale_factor; i++) {
            //     if (subtree_util[i].key != 0) {
            //         printf("%d:%d ", subtree_util[i].key, subtree_util[i].util);
            //     }
            // }
            // printf("\n");

            // count num primary
            int subtree_util_counter = 0;
            for (int i = 0; i < work_item.max_item * scale_factor; i++) {
                if (subtree_util[i].util >= min_util) {
                    subtree_util_counter++;
                }
            }

            // printf("Subtree Utility Counter: %d\n", subtree_util_counter);

            if (subtree_util_counter)
            {
                for (int i = 0; i < work_item.max_item * scale_factor; i++) {
                    if (subtree_util[i].util < min_util) continue;

                    WorkItem new_work_item;
                    new_work_item.pattern = n_pattern;
                    new_work_item.items = n_items;
                    new_work_item.num_items = new_item_counter;
                    new_work_item.start = n_start;
                    new_work_item.end = n_end;
                    new_work_item.utility = n_utility;
                    new_work_item.num_transactions = new_transaction_counter;
                    new_work_item.primary_item = subtree_util[i].key;
                    new_work_item.max_item = local_util_counter;
                    new_work_item.work_done = work_done;
                    new_work_item.work_count = subtree_util_counter;

                    while(!work_queue->enqueue(new_work_item)); // busy wait

                }
            }
















            printf("\n");
            int ret = atomicAdd(work_item.work_done, 1);
            if (ret == work_item.work_count - 1) {
                // printf("Work Done\n");
                memory_manager->free(work_item.pattern);
                memory_manager->free(work_item.work_done);
                memory_manager->free(work_item.utility);
                memory_manager->free(work_item.start);
                memory_manager->free(work_item.end);
                memory_manager->free(work_item.items);
            }

            atomicSub(&work_queue->work_count, 1);
        }
    }
}


int main(int argc, char* argv[]) {
    // Parse command-line arguments using args_parser
    ParsedArgs args;
    if (!parseArguments(argc, argv, args)) {
        // Parsing failed; exit the program
        return EXIT_FAILURE;
    }

    ReadFileResult fileResult = read_file(args.filename, args.separator, args.utility);
    // Note: Replace " " and 100 with actual `sep` and `minUtil` as needed

    // Access the parsed data
    auto& filteredTransactions = fileResult.filteredTransactions;
    auto& primary = fileResult.primary;
    auto& rename = fileResult.rename;
    int max_item = fileResult.max_item;

    // // (Optional) Display parsed data
    // std::cout << "Filtered Transactions:\n";
    // for (const auto& [key, val] : filteredTransactions) {
    //     for (int i = 0; i < key.size(); i++) {
    //         std::cout << key[i] << ":" << val[i] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "Primary Items:\n";
    // for (uint32_t item : primary) {
    //     std::cout << item << " ";
    // }
    // std::cout << "\n";



    // Flatten filteredTransactions
    std::vector<Item> items;
    std::vector<int> start;
    std::vector<int> end;

    for (const auto& [key, val] : filteredTransactions) {
        start.push_back(items.size());
        for (int i = 0; i < key.size(); i++) {
            items.push_back({key[i], val[i]});
        }
        end.push_back(items.size());
    }

    // copy items to device
    Item* d_items;
    int* d_start;
    int* d_end;
    int* d_primary;

    size_t num_items = items.size();
    size_t num_transactions = start.size();
    
    HANDLE_ERROR(cudaMalloc(&d_items, num_items * sizeof(Item)));
    HANDLE_ERROR(cudaMalloc(&d_start, num_transactions * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_end, num_transactions * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_primary, primary.size() * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(d_items, items.data(), num_items * sizeof(Item), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_start, start.data(), num_transactions * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_end, end.data(), num_transactions * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(int), cudaMemcpyHostToDevice));

    int32_t *d_high_utility_patterns;
    cudaMallocManaged(&d_high_utility_patterns, 1024ULL * 1024ULL * 1024ULL); // 1GB
    d_high_utility_patterns[1] = 2;


    using MemoryManagerType = OuroVAPQ;
    MemoryManagerType memory_manager;
	memory_manager.initialize(0, 1024 * 1024 * 1024);
    // memory_manager.initialize();


    // start work queue
    AtomicWorkQueue<WorkItem, work_queue_size> *work_queue;
    cudaMallocManaged(&work_queue, sizeof(AtomicWorkQueue<WorkItem, work_queue_size>));
    work_queue->init();

    // copy over to device
    initial_copy<<<1, 1>>>(memory_manager.getDeviceMemoryManager(), work_queue, d_items, d_start, d_end, d_primary, primary.size(), num_transactions, max_item);
    // Synchronize to ensure kernel completion
    HANDLE_ERROR(cudaDeviceSynchronize());

    // free the memory
    cudaFree(d_items);
    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_primary);


    // // get the first work item and print it
    // WorkItem work_item;
    // work_queue->host_dequeue(work_item);

    // std::cout << "Work Item:\n";
    // // print address of pattern
    // std::cout << "Pattern: " << work_item.pattern << "\n";
    // // print address of items
    // std::cout << "Items: " << work_item.items << "\n";

    initial_test<<<1,1>>>(memory_manager.getDeviceMemoryManager(), work_queue, args.utility, d_high_utility_patterns);
    // Synchronize to ensure kernel completion
    HANDLE_ERROR(cudaDeviceSynchronize());


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
                // high_utility_patterns_str += std::to_string(rename[high_utility_patten[j]]) + " ";
                high_utility_patterns_str += rename[high_utility_patten[j]] + " ";
                // std::cout << rename[high_utility_patten[j]] << " ";
            }
            Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];

            high_utility_patterns_str = "";
            high_utility_patten.clear();
        }
    // std::cout << high_utility_patterns_str << std::endl;
    // for (const auto &p : Patterns)
    // {
    //     std::cout << p.first << "UTIL: " << p.second << std::endl;
    // }

    std::cout << "High Utility Patterns: " << d_high_utility_patterns[0] << "\n";




    return 0;

}