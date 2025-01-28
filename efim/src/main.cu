
#include "allocator.cuh"

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define BUCKET_SCALE 3
#define WORK_QUEUE_CAPACITY 8192

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#include <assert.h>
#define cdpErrchk(ans)                        \
    {                                         \
        cdpAssert((ans), __FILE__, __LINE__); \
    }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            assert(0);
    }
}

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

__device__ uint32_t array_hasher(Item *array, uint32_t size)
{
    uint32_t hash = 0;
    for (uint32_t i = 0; i < size; i++)
    {
        hash = pcg_hash(hash + array[i].id);
    }
    return hash;
}

__device__ uint32_t array_hasher_function(Item *array, uint32_t size, uint32_t tableSize)
{
    uint32_t hash = array_hasher(array, size);
    return hash % tableSize;
}

__device__ int binarySearchItems(const Item *items, int n, uint32_t search_id, int offset, int length)
{
    // Validate that the provided range is within bounds.
    if (offset < 0 || offset >= n || length <= 0 || (offset + length) > n)
    {
        return -1;
    }

    int l = offset;
    int r = offset + length - 1;
    while (l <= r)
    {
        int mid = l + (r - l) / 2;
        uint32_t mid_id = items[mid].id;
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

// Helper: split a string by a delimiter.
std::vector<std::string> split(const std::string &str, const std::string &delimiter)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        tokens.push_back(str.substr(prev, pos - prev));
        prev = pos + delimiter.length();
    }
    tokens.push_back(str.substr(prev));
    return tokens;
}


// mine_kernel_buffer<<<1, 1>>>(alloc, workQueue, min_util, d_high_utility_patterns);
__global__ void mine_kernel_buffer(BumpAllocator *alloc, AtomicWorkQueue<WorkItem, WORK_QUEUE_CAPACITY> *workQueue, uint32_t minUtil, uint32_t *high_utility_patterns)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // get the workitem from the work queue and print number of primary
    WorkItem wi;
    // int wk = workQueue->get_work_count();
    while (workQueue->get_work_count() > 0)
    {
        if (!workQueue->dequeue(wi)) continue;
        printf("Acquired work item with %d primary items.\n", wi.num_primary);



        // uint32_t local_util_count = workItem->maxItem * 3;
        uint32_t scratch_bytes = sizeof(Item) * wi.num_items +
                                 sizeof(uint32_t) * wi.num_transactions * 3 +
                                 sizeof(Item) * wi.max_item * BUCKET_SCALE * 2;

        void *scratch = bump_alloc(alloc, scratch_bytes);

        if (!scratch)
        {
            printf("Error allocating scratch pad.\n");
            return;
        }

        Item *scratch_items = (Item *)scratch;
        scratch += sizeof(Item) * wi.num_items;

        uint32_t *scratch_start = (uint32_t *)scratch;
        scratch += sizeof(uint32_t) * wi.num_transactions;
        uint32_t *scratch_end = (uint32_t *)scratch;
        scratch += sizeof(uint32_t) * wi.num_transactions;
        uint32_t *scratch_utility = (uint32_t *)scratch;
        scratch += sizeof(uint32_t) * wi.num_transactions;

        Item *local_util = (Item *)scratch;
        scratch += sizeof(Item) * wi.max_item * BUCKET_SCALE;
        Item *subtree_util = (Item *)scratch;

        // printf("Num Primary: %d\n", wi.num_primary);
        for (int i = 0; i < wi.num_primary; i++)
        {
            uint32_t item = wi.primary[i];
            // printf("Item: %d\n", item);

            // reset local utility
            for (int j = 0; j < wi.max_item * BUCKET_SCALE; j++)
            {
                local_util[j].id = 0;
                local_util[j].utility = 0;

                subtree_util[j].id = 0;
                subtree_util[j].utility = 0;
            }

            // find the item in the transactions
            uint32_t item_counter = 0;
            uint32_t transaction_counter = 0;
            uint32_t pattern_utility = 0;

            for (int j = 0; j < wi.num_transactions; j++)
            {
                //  if start == end continue
                if (wi.start[j] == wi.end[j])
                {
                    continue;
                }
                int idx = binarySearchItems(wi.items, wi.num_items, item, wi.start[j], wi.end[j] - wi.start[j]);
                uint32_t temp_util = 0;

                if (idx != -1)
                {
                    scratch_start[transaction_counter] = item_counter;
                    scratch_utility[transaction_counter] = wi.items[idx].utility + wi.utility[j];
                    pattern_utility += scratch_utility[transaction_counter];
                    temp_util = scratch_utility[transaction_counter];

                    for (int k = idx + 1; k < wi.end[j]; k++)
                    {
                        // printf("%d:%d ", wi.items[k].id, wi.items[k].utility);
                        scratch_items[item_counter] = wi.items[k];
                        temp_util += wi.items[k].utility;
                        item_counter++;
                    }
                    // printf("|| %d ||%d\n", scratch_utility[transaction_counter], temp_util);

                    if (item_counter != scratch_start[transaction_counter])
                    {
                        scratch_end[transaction_counter] = item_counter;
                        transaction_counter++;
                    }

                    for (int k = idx + 1; k < wi.end[j]; k++)
                    {
                        // we do a hash of the item id to find where to put it in the local utility
                        uint32_t hash = hashFunction(wi.items[k].id, wi.max_item * BUCKET_SCALE);
                        while (true)
                        {
                            if (local_util[hash].id == 0 || local_util[hash].id == wi.items[k].id)
                            {
                                local_util[hash].id = wi.items[k].id;
                                local_util[hash].utility += temp_util;
                                break;
                            }
                            else
                            {
                                // linear probing
                                hash = (hash + 1) % wi.max_item * BUCKET_SCALE;
                            }
                        }
                    }
                }
            }

            uint32_t *n_pattern;

            // allocate new pattern if utility is greater than minUtil or transaction counter is greater than 1
            if (pattern_utility >= minUtil || transaction_counter)
            {
                n_pattern = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * (wi.pattern[0] + 3));
                if (!n_pattern)
                {
                    printf("Error allocating new pattern.\n");
                    return;
                }

                memcpy(n_pattern, wi.pattern, sizeof(uint32_t) * (wi.pattern[0] + 1));
                n_pattern[0] += 1;
                n_pattern[n_pattern[0]] = item;

                // // print pattern
                // printf("Pattern: ");
                // for (int j = 0; j < n_pattern[0]; j++)
                // {
                //     printf("%d ", n_pattern[j + 1]);
                // }
                // printf(": %d\n", pattern_utility);
            }

            if (pattern_utility >= minUtil)
            {
                // add pattern to high utility patterns [0 is number of patterns][1 is offset to start wiritn from]
                uint32_t ret = atomicAdd(&high_utility_patterns[0], 1);
                printf("Pat count: %d\tTID: %d\tMemUsed: %llu\n", ret, tid, alloc->offset);
                uint32_t offset = atomicAdd(&high_utility_patterns[1], n_pattern[0] + 2); // 1 for utilty and 1 spacer
                for (int j = 0; j < n_pattern[0]; j++)
                {
                    high_utility_patterns[offset + j] = n_pattern[j + 1];
                }
                high_utility_patterns[offset + n_pattern[0]] = pattern_utility;
            }

            if (transaction_counter)
            {
                // count number of local utility that is greater than minUtil
                // printf("Transaction Counter: %d\n", transaction_counter);
                uint32_t local_util_counter = 0;
                for (int j = 0; j < wi.max_item * BUCKET_SCALE; j++)
                {
                    if (local_util[j].utility >= minUtil)
                    {
                        local_util_counter++;
                    }
                }

                uint32_t total_bytes_req = sizeof(Item) * item_counter +
                                           sizeof(uint32_t) * transaction_counter * 3 +
                                           sizeof(uint32_t) * local_util_counter;
                void *projection_ptr = bump_alloc(alloc, total_bytes_req);
                if (!projection_ptr)
                {
                    printf("Error allocating projection.\n");
                    return;
                }
                // printf("Transaction Counter: %d\n", transaction_counter);

                Item *projection_items = (Item *)projection_ptr;
                projection_ptr += sizeof(Item) * item_counter;
                uint32_t *projection_start = (uint32_t *)projection_ptr;
                projection_ptr += sizeof(uint32_t) * transaction_counter;
                uint32_t *projection_end = (uint32_t *)projection_ptr;
                projection_ptr += sizeof(uint32_t) * transaction_counter;
                uint32_t *projection_utility = (uint32_t *)projection_ptr;
                projection_ptr += sizeof(uint32_t) * transaction_counter;
                uint32_t *n_primary = (uint32_t *)projection_ptr;

                uint32_t new_item_count = 0;
                uint32_t new_transaction_count = 0;


                // // allocate for hash table. we reuse the itemstruct but id is hash and util is trans id
                // Item *hash_table = (Item *)bump_alloc(alloc, sizeof(Item) * transaction_counter * BUCKET_SCALE);
                // // set all hash table to 0
                // for (int j = 0; j < transaction_counter; j++)
                // {
                //     hash_table[j].id = 0;
                //     hash_table[j].utility = 0;
                // }




                for (int j = 0; j < wi.num_transactions; j++)
                {
                    if (scratch_start[j] != scratch_end[j])
                    {
                        // printf("j: %d\n", j);
                        projection_start[new_transaction_count] = new_item_count;

                        for (int k = scratch_start[j]; k < scratch_end[j]; k++)
                        {
                            // check if the item is in the local utility and if it is greater than minUtil
                            uint32_t hash = hashFunction(scratch_items[k].id, wi.max_item * 3);
                            while (true)
                            {
                                if (local_util[hash].id == scratch_items[k].id)
                                {
                                    if (local_util[hash].utility >= minUtil)
                                    {
                                        projection_items[new_item_count] = scratch_items[k];
                                        new_item_count++;
                                    }
                                    break;
                                }
                                else
                                {
                                    hash = (hash + 1) % wi.max_item * 3;
                                }
                            }
                        }

                        uint32_t hash = array_hasher(scratch_items + scratch_start[j], scratch_end[j] - scratch_start[j]);
                        uint32_t hash_idx = hash % transaction_counter * BUCKET_SCALE;
                        // uint32_t hash_idx = array_hasher_function(scratch_items + scratch_start[j], scratch_end[j] - scratch_start[j], transaction_counter * BUCKET_SCALE);
                        // // if hash is not in the hash table, add it
                        // // if hash is in the hash table, check if the items.id match
                        //     // if they match, add the utility and remove
                        // // else linear probe and add

                        



                        projection_end[new_transaction_count] = new_item_count;
                        projection_utility[new_transaction_count] = scratch_utility[j];

                        new_transaction_count++;

                        if (new_transaction_count == transaction_counter)
                        {
                            break;
                        }
                    }
                }

                // iterate through the projected transactions
                for (int j = 0; j < new_transaction_count; j++)
                {
                    uint32_t temp_util = projection_utility[j];
                    for (int k = projection_start[j]; k < projection_end[j]; k++)
                    {   
                        temp_util += projection_items[k].utility;
                    }
                    uint32_t temp = 0;
                    for (int k = projection_start[j]; k < projection_end[j]; k++)
                    {
                        uint32_t hash = hashFunction(projection_items[k].id, wi.max_item * 3);
                        while (true)
                        {
                            if (subtree_util[hash].id == projection_items[k].id || subtree_util[hash].id == 0)
                            {
                                subtree_util[hash].id = projection_items[k].id;
                                subtree_util[hash].utility += temp_util - temp;
                                temp += projection_items[k].utility;
                                break;
                            }
                            else
                            {
                                hash = (hash + 1) % wi.max_item * 3;
                            }
                        }
                    }
                }


                // printf("new_transaction_count: %d\n", new_transaction_count);

                uint32_t primary_count = 0;
                for (int j = 0; j < wi.max_item * 3; j++)
                {
                    if (subtree_util[j].id == 0)
                        continue;
                    // printf("%d:%d ", local_util[j].id, local_util[j].utility);
                    if (subtree_util[j].utility >= minUtil)
                    {
                        n_primary[primary_count] = subtree_util[j].id;
                        primary_count++;
                    }
                }

                // printf("Primary Count: %d\n", primary_count);

                if (primary_count)
                {
                    WorkItem this_thing;

                    this_thing.pattern = n_pattern;
                    this_thing.items = projection_items;
                    this_thing.num_items = new_item_count;
                    this_thing.start = projection_start;
                    this_thing.end = projection_end;
                    this_thing.utility = projection_utility;
                    this_thing.num_transactions = new_transaction_count;
                    this_thing.primary = n_primary;
                    this_thing.num_primary = primary_count;
                    this_thing.max_item = local_util_counter;
                    while (!workQueue->enqueue(this_thing))
                    {
                        printf("Waiting for space in the queue.\n");
                    }
                }
            }

        //     // printf("\n");
        }

        uint32_t old = atomicSub(&workQueue->work_count, 1);
    }
}

class cuEFIM
{
public:
    cuEFIM(const std::string &iFile, int minUtil, const std::string &sep = "\t", uint64_t alloc_size = 4ULL * GIGA) : inputFile(iFile), minUtil(minUtil), runtime(0), alloc_size(alloc_size)
    {
        if (sep == "\\s")
            this->sep = " ";
        else if (sep == "\\t")
            this->sep = "\t";
        else
            this->sep = sep;
    }

    void mine()
    {



        auto start_time = std::chrono::high_resolution_clock::now();

        auto [transactions, primary] = _read_file();

        std::cout << "Time to read file: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;

        std::cout << "Number of transactions: " << transactions.size() << std::endl;
        std::cout << "Number of items: " << rename.size() << std::endl;
        std::cout << "Max item id: " << max_item << std::endl;

        // // print all transactions
        // std::cout << "Transactions: " << std::endl;
        // for (const auto &t : transactions)
        // {
        //     for (int i = 0; i < t.first.size(); i++)
        //     {
        //         std::cout << t.first[i] << ":" << t.second.vals[i] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // // print all primary
        // std::cout << "Primary: " << std::endl;
        // for (const auto &p : primary)
        // {
        //     std::cout << p << " ";
        // }
        // std::cout << std::endl;

        // prepare indices for device memory
        std::vector<uint32_t> start;
        std::vector<uint32_t> end;
        std::vector<Item> items;
        std::vector<uint32_t> util(transactions.size(), 0);

        uint32_t counter = 0;
        for (const auto &t : transactions)
        {
            start.push_back(counter);
            for (size_t i = 0; i < t.first.size(); ++i)
            {
                items.push_back(Item{static_cast<uint32_t>(t.first[i]), static_cast<uint32_t>(t.second[i])});
                counter++;
            }
            end.push_back(counter);
        }

        // create BumpAllocator
        std::cout << "Allocating " << alloc_size << " bytes for the bump allocator." << std::endl;
        BumpAllocator *alloc = createUnifiedBumpAllocator(alloc_size);
        if (!alloc)
        {
            std::cerr << "Error creating bump allocator." << std::endl;
            return;
        }

        // print alloc bytes used
        double mb_used = static_cast<double>(alloc->offset) / static_cast<double>(MEGA);
        std::cout << "Bytes used by the allocator: " << alloc->offset << "\t(MB: " << mb_used << ")" << std::endl;

        uint32_t pattern_init = 0;
        uint32_t *d_pattern = bump_allocate_and_copy(alloc, &pattern_init, 1);

        Item *d_items = bump_allocate_and_copy(alloc, items.data(), items.size());
        uint32_t *d_start = bump_allocate_and_copy(alloc, start.data(), start.size());
        uint32_t *d_end = bump_allocate_and_copy(alloc, end.data(), end.size());
        uint32_t *d_utility = bump_allocate_and_copy(alloc, util.data(), util.size());

        uint32_t *d_primary = bump_allocate_and_copy(alloc, primary.data(), primary.size());

        mb_used = static_cast<double>(alloc->offset) / static_cast<double>(MEGA);
        std::cout << "Bytes used by the allocator: " << alloc->offset << "\t(MB: " << mb_used << ")" << std::endl;

        if (!d_pattern || !d_start || !d_end || !d_primary || !d_items || !d_utility)
        {
            std::cerr << "Error allocating device memory." << std::endl;
            return;
        }

        uint32_t *d_high_utility_patterns;
        cudaMallocManaged(&d_high_utility_patterns, GIGA);
        d_high_utility_patterns[1] = 2;

        // Copy the data to device memory.
        cudaMemcpy(d_start, start.data(), start.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_end, end.data(), end.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_items, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice);
        cudaMemset(d_utility, 0, start.size() * sizeof(uint32_t));

        std::cout << "Time to allocate and copy data to device: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;

        AtomicWorkQueue<WorkItem, WORK_QUEUE_CAPACITY> *workQueue;
        cudaMallocManaged(&workQueue, sizeof(AtomicWorkQueue<WorkItem, WORK_QUEUE_CAPACITY>));

        // copy work items to device
        WorkItem initial;
        initial.pattern = d_pattern;
        initial.items = d_items;
        initial.num_items = items.size();
        initial.start = d_start;
        initial.end = d_end;
        initial.utility = d_utility;
        initial.num_transactions = transactions.size();
        initial.primary = d_primary;
        initial.num_primary = primary.size();
        initial.max_item = max_item;

        // initialize work queue
        workQueue->init();

        // enqueue initial work item
        workQueue->host_enqueue(initial);

        // print work count
        printf("Work Count: %d\n", workQueue->work_count);

        mine_kernel_buffer<<<64, 1>>>(alloc, workQueue, minUtil, d_high_utility_patterns);
        cudaDeviceSynchronize();
        std::cout << "Work Count: " << workQueue->work_count << std::endl;

        // while (workQueue->work_count > 0)
        // {
        //     mine_kernel_buffer<<<64, 1>>>(alloc, workQueue, minUtil, d_high_utility_patterns);
        //     cudaDeviceSynchronize();
        //     std::cout << "Work Count: " << workQueue->work_count << std::endl;
        //     std::cout << "Patterns: " << d_high_utility_patterns[0] << std::endl;   
        //     // print pattern of WorkItem at the top of the queue
            
        // }

        runtime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

        mb_used = static_cast<double>(alloc->offset) / static_cast<double>(MEGA);
        std::cout << "Bytes used by the allocator: " << alloc->offset << "\t(MB: " << mb_used << ")" << std::endl;

        // free device memory
        freeUnifiedBumpAllocator(alloc);
        printf("Number of high utility patterns: %d\n", d_high_utility_patterns[0]);

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
                high_utility_patterns_str += std::to_string(high_utility_patten[j]) + " ";
            }
            Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];

            high_utility_patterns_str = "";
            high_utility_patten.clear();
        }
        // std::cout << high_utility_patterns_str << std::endl;
        for (const auto &p : Patterns)
        {
            std::cout << p.first << "UTIL: " << p.second << std::endl;
        }
    }

    // Print final results.
    void printResults() const
    {
        std::cout << "Total number of High Utility Patterns: " << Patterns.size() << "\n";
        std::cout << "Total Execution Time (seconds): " << runtime / 1000 << "\n";
        // Memory info is platform‐dependent in C++ so is omitted here.
    }

    // Get discovered patterns.
    const std::map<std::string, int> &getPatterns() const
    {
        return Patterns;
    }

    double getRuntime() const { return runtime; }

    // Save results to a file.
    void save(const std::string &outFile)
    {
        std::ofstream writer(outFile);
        if (!writer)
        {
            std::cerr << "Error opening output file: " << outFile << "\n";
            return;
        }
        // Here we write the discovered patterns (stored in Patterns)
        for (const auto &entry : Patterns)
        {
            writer << entry.first << ":" << entry.second << "\n";
        }
        writer.close();
    }

private:
    std::string inputFile;
    std::string sep;
    uint32_t minUtil;
    uint64_t alloc_size;

    uint32_t max_item;

    std::unordered_map<uint32_t, std::string> rename;

    std::map<std::string, int> Patterns;

    double runtime;

    void _increase_cuda_stack()
    {
        size_t currentStackSize;
        cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
        printf("Current device stack size: %zu bytes\n", currentStackSize);

        // cudaDeviceSetLimit(cudaLimitStackSize, 8192);
        size_t stackSize = 64 * KILO; // 8192 per thread (adjust as needed)
        cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to set device stack size (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
        printf("Current device stack size: %zu bytes\n", currentStackSize);
    }

    std::pair<std::map<std::vector<int>, std::vector<int>>, std::vector<uint32_t>> _read_file()
    {
        // File data: each entry is a pair: (vector of item strings, vector of utility ints)
        std::vector<std::pair<std::vector<std::string>, std::vector<int>>> fileData;
        // TWU dictionary: item string -> total weight
        std::unordered_map<std::string, int> twu;

        std::ifstream infile(inputFile);
        if (!infile)
        {
            std::cerr << "Error opening file: " << inputFile << "\n";
            exit(1);
        }
        std::string line;
        while (std::getline(infile, line))
        {
            // Expected format: items : weight : utility_list
            // Items and utility_list are further separated by sep.
            std::vector<std::string> parts = split(line, ":");
            if (parts.size() < 3)
                // continue;
                std::cerr << "Invalid line: " << line << "\n";

            std::vector<std::string> items = split(parts[0], sep);

            int weight = std::stoi(parts[1]);
            std::vector<std::string> utilStrs = split(parts[2], sep);

            std::vector<int> utils;
            for (const auto &s : utilStrs)
                utils.push_back(std::stoi(s));
            fileData.push_back({items, utils});
            // Update TWU: add weight for each item
            for (const auto &item : items)
            {
                twu[item] += weight;
            }
        }
        infile.close();

        // // print all twu
        // std::cout << "TWU: " << std::endl;
        // for (const auto &t : twu) {
        //     std::cout << t.first << ":" << t.second << std::endl;
        // }

        // Filter twu based on minUtil threshold.
        for (auto it = twu.begin(); it != twu.end();)
        {
            if (it->second < minUtil)
                it = twu.erase(it);
            else
                ++it;
        }

        // Create a sorted vector of (item, utility) in descending order by utility.
        std::vector<std::pair<std::string, int>> sortedTWU(twu.begin(), twu.end());
        std::sort(sortedTWU.begin(), sortedTWU.end(),
                  [](const auto &a, const auto &b)
                  { return a.second > b.second; });

        // Map each item (string) to an integer (starting from count downwards).
        std::unordered_map<std::string, int> strToInt;
        int t = static_cast<int>(sortedTWU.size());
        max_item = t;
        for (const auto &p : sortedTWU)
        {
            strToInt[p.first] = t;
            rename[t] = p.first;
            t--;
        }

        // Build filtered transactions and compute subtree utility.
        std::unordered_map<int, int> subtree;
        std::map<std::vector<int>, std::vector<int>> filteredTransactions;
        for (const auto &entry : fileData)
        {
            const std::vector<std::string> &items = entry.first;
            const std::vector<int> &utils = entry.second;
            std::vector<std::pair<int, int>> transaction;
            for (size_t i = 0; i < items.size(); ++i)
            {
                if (strToInt.find(items[i]) != strToInt.end())
                {
                    transaction.push_back({strToInt[items[i]], utils[i]});
                }
            }
            if (transaction.empty())
                continue;
            // Sort transaction by item id.
            std::sort(transaction.begin(), transaction.end(),
                      [](const std::pair<int, int> &a, const std::pair<int, int> &b)
                      { return a.first < b.first; });

            std::vector<int> key;
            std::vector<int> val;
            for (const auto &p : transaction)
            {
                key.push_back(p.first);
                val.push_back(p.second);
            }

            // Use key (sorted vector) as key for filteredTransactions.
            if (filteredTransactions.find(key) == filteredTransactions.end())
            {
                // filteredTransactions[key] = Transaction{key, val, 0};
                filteredTransactions[key] = val;
            }
            else
            {
                // If the transaction already exists, add corresponding utilities.
                // Transaction &trans = filteredTransactions[key];
                for (size_t i = 0; i < val.size(); ++i)
                    // trans.vals[i] += val[i];
                    filteredTransactions[key][i] += val[i];
            }

            // Compute subtree utility for this transaction.
            int subUtil = 0;
            for (int v : val)
                subUtil += v;
            int temp = 0;
            for (size_t i = 0; i < key.size(); ++i)
            {
                subtree[key[i]] += subUtil - temp;
                temp += val[i];
            }
        }

        // Determine primary items: those with subtree utility >= minUtil.
        std::vector<uint32_t> primary;
        for (const auto &p : subtree)
        {
            if (p.second >= minUtil)
                primary.push_back(p.first);
        }

        return std::make_pair(filteredTransactions, primary);
    }
};

int main(int argc, char *argv[])
{
    if (argc < 4 || argc > 6)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <delimiter> <minUtil> [<alloc_size>] [<output_file>]" << std::endl;
        return 1;
    }

    std::string filename(argv[1]);
    std::string delimiter(argv[2]);
    uint32_t minUtil;

    try
    {
        minUtil = std::stoi(argv[3]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    if (argc == 5)
    {
        cuEFIM efim(filename, minUtil, delimiter, std::stoull(argv[4]) * GIGA);
        efim.mine();
        efim.printResults();
    }
    else if (argc == 6)
    {
        cuEFIM efim(filename, minUtil, delimiter, std::stoull(argv[4]) * GIGA);
        efim.mine();
        efim.printResults();
        efim.save(argv[5]);
    }
    else
    {
        cuEFIM efim(filename, minUtil, delimiter);
        efim.mine();
        efim.printResults();
    }

    return 0;
}
