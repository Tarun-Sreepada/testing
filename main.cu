
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "args.h"   // Include the args parser
#include "parser.h" // Include the file reader
#include "work.cuh" // Include the work queue

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "Utility.h"

#define scale 2

// Binary Search Utility
int binarySearchItems(const Item *items, int n, uint32_t search_id, int offset, int length)
{
    if (offset < 0 || offset >= n || length <= 0 || (offset + length) > n)
        return -1;
    int l = offset, r = offset + length - 1;
    while (l <= r)
    {
        int mid = l + (r - l) / 2;
        if (items[mid].key == search_id)
            return mid;
        items[mid].key < search_id ? l = mid + 1 : r = mid - 1;
    }
    return -1;
}

uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint32_t hashFunction(uint32_t key, uint32_t tableSize)
{
    return pcg_hash(key) % tableSize;
}

uint32_t items_hasher(const Item *items, int n, int tableSize)
{
    uint32_t hash = 0;
    for (int i = 0; i < n; i++)
        hash ^= pcg_hash(items[i].key);
    return hash % tableSize;
}

template <typename MemoryManagerType>
__global__ void copy(
    MemoryManagerType *memory_manager,
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

    int bytes_to_allocate = 4 * sizeof(int) +                // pattern
                            item_count * sizeof(Item) +      // items
                            num_transactions * sizeof(int) + // start
                            num_transactions * sizeof(int) + // end
                            num_transactions * sizeof(int);  // utility
    2 * sizeof(int);                                         // work_done

    void *base_ptr = memory_manager->malloc(bytes_to_allocate);
    memset(base_ptr, 0, bytes_to_allocate);

    void *ptr = base_ptr;
    int *pattern = reinterpret_cast<int *>(ptr);
    ptr += 4 * sizeof(int);
    Item *n_items = reinterpret_cast<Item *>(ptr);
    ptr += item_count * sizeof(Item);
    int *n_start = reinterpret_cast<int *>(ptr);
    ptr += num_transactions * sizeof(int);
    int *n_end = reinterpret_cast<int *>(ptr);
    ptr += num_transactions * sizeof(int);
    int *n_utility = reinterpret_cast<int *>(ptr);
    ptr += num_transactions * sizeof(int);
    int *work_done = reinterpret_cast<int *>(ptr);
    ptr += sizeof(int);

    pattern[0] = 0;
    work_done[0] = 0;
    memcpy(n_start, start, num_transactions * sizeof(int));
    memcpy(n_end, end, num_transactions * sizeof(int));
    memcpy(n_items, items, item_count * sizeof(Item));

    printf("DB: \n");
    for (int i = 0; i < num_transactions; i++)
    {
        printf("%d|", n_utility[i]);
        for (int j = n_start[i]; j < n_end[i]; j++)
        {
            printf("%d:%d ", n_items[j].key, n_items[j].util);
        }
        printf("\n");
    }

    for (int i = 0; i < num_primary; i++)
    {
        WorkItem work_item;
        work_item.pattern = pattern;
        work_item.items = n_items;
        work_item.num_items = item_count;
        work_item.start = n_start;
        work_item.end = n_end;
        work_item.utility = n_utility;
        work_item.num_transactions = num_transactions;
        work_item.primary_item = primary[i];
        work_item.max_item = max_item;
        work_item.work_done = work_done;
        work_item.work_count = num_primary;
        work_item.bytes = bytes_to_allocate;
        work_item.base_ptr = base_ptr;

        stack_push(work_queue, work_item);
    }
}

template <typename MemoryManagerType>
__global__ void mine(MemoryManagerType *memory_manager, AtomicWorkStack *work_queue, int min_util, int *high_utility_patterns)
{
    WorkItem work_item;

    while (work_queue->work_count)
    {
        if (stack_pop(work_queue, &work_item))
        {
            printf("Item: %d\n", work_item.primary_item);

            printf("DB: \n");
            for (int i = 0; i < work_item.num_transactions; i++)
            {
                for (int j = work_item.start[i]; j < work_item.end[i]; j++)
                {
                    printf("%d:%d ", work_item.items[j].key, work_item.items[j].util);
                }
                printf("\n");
            }





            printf("\n");
            atomicSub(&work_queue->work_count, 1);
        }
    }
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

    ReadFileResult fileResult = read_file(args.filename, args.separator, args.utility);
    // Note: Replace " " and 100 with actual `sep` and `minUtil` as needed

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

    // copy items to device
    Item *d_items;
    int *d_start;
    int *d_end;
    int *d_primary;

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
    MemoryManagerType mm;
    mm.initialize(0, 1024 * 1024 * 1024);
    // memory_manager.initialize();

    // start work queue
    // AtomicWorkQueue<WorkItem, work_queue_size> *work_queue;
    AtomicWorkStack *work_queue;
    cudaMalloc(&work_queue, sizeof(AtomicWorkStack));

    copy<<<1, 1>>>(mm.getDeviceMemoryManager(), work_queue, d_items, d_start, d_end, d_primary, primary.size(), num_transactions, max_item);
    // Synchronize to ensure kernel completion
    HANDLE_ERROR(cudaDeviceSynchronize());

    // free the memory
    cudaFree(d_items);
    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_primary);

    mine<<<1, 1>>>(mm.getDeviceMemoryManager(), work_queue, args.utility, d_high_utility_patterns);
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
            high_utility_patterns_str += rename[high_utility_patten[j]] + " ";
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