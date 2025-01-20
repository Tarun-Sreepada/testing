#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <stdint.h>
#include <cuda_runtime.h>
#include <chrono>
#include <unordered_map>
#include <map>
#include <fstream>
#include <algorithm>
#include "allocator.cuh"

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA

#define BUCKET_SCALE 3

struct Item
{
    uint32_t id;
    uint32_t utility;
};

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

// A structure to hold transactions.
struct Transaction
{
    std::vector<int> keys; // sorted items (represented by int)
    std::vector<int> vals; // corresponding utility values
    int acc;               // accumulated utility
};

// For ordering keys in maps (we use vector<int> as a key).
struct VectorComparator
{
    bool operator()(const std::vector<int> &a, const std::vector<int> &b) const
    {
        return a < b; // lexicographical compare provided by std::vector
    }
};



__global__ void mine_kernel(BumpAllocator *alloc, uint32_t *pattern, Item *items, uint32_t nItems,
                            uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
                            uint32_t *primary, uint32_t numPrimary, uint32_t maxItem,
                            uint32_t minUtil, uint32_t *high_utility_patterns)
{
    // create scratch pad

    // we need items, start, end, utility, local
    // items + 3x transaction + maxItem * 3 (bucketSize)
    uint32_t local_util_count = maxItem * 3;

    uint32_t scratch_bytes = sizeof(Item) * nItems +
                             sizeof(uint32_t) * nTransactions * 3 +
                             sizeof(Item) * local_util_count * 2;

    void *scratch = bump_alloc(alloc, scratch_bytes);

    if (!scratch)
    {
        printf("Error allocating scratch pad.\n");
        return;
    }

    Item *scratch_items = (Item *)scratch;
    scratch += sizeof(Item) * nItems;

    uint32_t *scratch_start = (uint32_t *)scratch;
    scratch += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_end = (uint32_t *)scratch;
    scratch += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_utility = (uint32_t *)scratch;
    scratch += sizeof(uint32_t) * nTransactions;

    Item *local_util = (Item *)scratch;
    scratch += sizeof(Item) * local_util_count;
    Item *subtree_util = (Item *)scratch;

    for (int i = 0; i < numPrimary; i++)
    {
        uint32_t item = primary[i];
        // printf("Item: %d\n", item);

        // reset local utility
        for (int j = 0; j < local_util_count; j++)
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

        for (int j = 0; j < nTransactions; j++)
        {
            int idx = binarySearchItems(items, nItems, item, start[j], end[j] - start[j]);
            uint32_t temp_util = 0;

            if (idx != -1)
            {
                scratch_start[transaction_counter] = item_counter;
                scratch_utility[transaction_counter] = items[idx].utility + utility[j];
                pattern_utility += scratch_utility[transaction_counter];
                temp_util = scratch_utility[transaction_counter];

                for (int k = idx + 1; k < end[j]; k++)
                {
                    // printf("%d:%d ", items[k].id, items[k].utility);
                    scratch_items[item_counter] = items[k];
                    temp_util += items[k].utility;
                    item_counter++;
                }
                // printf("|| %d ||%d\n", scratch_utility[transaction_counter], temp_util);

                if (item_counter != scratch_start[transaction_counter])
                {
                    scratch_end[transaction_counter] = item_counter;
                    transaction_counter++;
                }

                for (int k = idx + 1; k < end[j]; k++)
                {
                    // we do a hash of the item id to find where to put it in the local utility
                    uint32_t hash = hashFunction(items[k].id, local_util_count);
                    while (true)
                    {
                        if (local_util[hash].id == 0 || local_util[hash].id == items[k].id)
                        {
                            local_util[hash].id = items[k].id;
                            local_util[hash].utility += temp_util;
                            break;
                        }
                        else
                        {
                            // linear probing
                            hash = (hash + 1) % local_util_count;
                        }
                    }
                }
            }
        }
        // printf("Pattern Utility: %d\n", pattern_utility);

        // // print all local utility
        // printf("Local Utility: \n");
        // for (int j = 0; j < local_util_count; j++)
        // {
        //     if (local_util[j].id != 0)
        //     {
        //         printf("%d:%d ", local_util[j].id, local_util[j].utility);
        //     }
        // }
        // printf("\n");

        uint32_t *n_pattern;

        // allocate new pattern if utility is greater than minUtil or transaction counter is greater than 1
        if (pattern_utility >= minUtil || transaction_counter)
        {
            n_pattern = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * (pattern[0] + 3));
            if (!n_pattern)
            {
                printf("Error allocating new pattern.\n");
                return;
            }

            memcpy(n_pattern, pattern, sizeof(uint32_t) * (pattern[0] + 1));
            n_pattern[0] += 1;
            n_pattern[n_pattern[0]] = item;

            // // print pattern
            // printf("Pattern: ");
            // for (int j = 0; j < n_pattern[0]; j++)
            // {
            //     printf("%d ", n_pattern[j + 1]);
            // }
            // printf("\n");
        }

        if (pattern_utility >= minUtil)
        {
            // add pattern to high utility patterns [0 is number of patterns][1 is offset to start wiritn from]
            atomicAdd(&high_utility_patterns[0], 1);
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
            uint32_t local_util_counter = 0;
            for (int j = 0; j < local_util_count; j++)
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

            for (int j = 0; j < nTransactions; j++)
            {
                if (scratch_start[j] != scratch_end[j])
                {
                    projection_start[new_transaction_count] = new_item_count;
                    uint32_t temp_util = scratch_utility[j];

                    for (int k = scratch_start[j]; k < scratch_end[j]; k++)
                    {
                        // check if the item is in the local utility and if it is greater than minUtil
                        uint32_t hash = hashFunction(scratch_items[k].id, local_util_count);
                        while (true)
                        {
                            if (local_util[hash].id == scratch_items[k].id)
                            {
                                if (local_util[hash].utility >= minUtil)
                                {
                                    projection_items[new_item_count] = scratch_items[k];
                                    temp_util += scratch_items[k].utility;
                                    new_item_count++;
                                }
                                break;
                            }
                            else
                            {
                                hash = (hash + 1) % local_util_count;
                            }
                        }
                    }
                    if (new_item_count != scratch_start[j])
                    {
                        projection_end[new_transaction_count] = new_item_count;
                        projection_utility[new_transaction_count] = scratch_utility[j];
                        new_transaction_count++;
                    }

                    // calc subtree utility
                    uint32_t temp = 0;
                    for (int k = projection_start[new_transaction_count - 1]; k < projection_end[new_transaction_count - 1]; k++)
                    {
                        uint32_t hash = hashFunction(projection_items[k].id, local_util_count);
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
                                hash = (hash + 1) % local_util_count;
                            }
                        }
                    }

                    if (new_transaction_count == transaction_counter)
                    {
                        break;
                    }

                    // projection_end[new_transaction_count] = new_item_count;
                    // new_transaction_count++;
                }
            }

            // // print projected DB
            // printf("Projected DB: \n");
            // for (int j = 0; j < transaction_counter; j++)
            // {
            //     for (int k = scratch_start[j]; k < scratch_end[j]; k++)
            //     {
            //         printf("%d:%d ", scratch_items[k].id, scratch_items[k].utility);
            //     }
            //     printf("|| %d\n", scratch_utility[j]);
            // }

            // // print all subtree utility
            // printf("Subtree Utility: \n");
            // for (int j = 0; j < local_util_count; j++)
            // {
            //     if (subtree_util[j].id != 0)
            //     {
            //         printf("%d:%d ", subtree_util[j].id, subtree_util[j].utility);
            //     }
            // }

            // printf("\n");

            uint32_t primary_count = 0;
            for (int j = 0; j < local_util_count; j++)
            {
                if (subtree_util[j].utility >= minUtil)
                {
                    n_primary[primary_count] = subtree_util[j].id;
                    primary_count++;
                }
            }

            // // print all primary
            // printf("Primary: \n");
            // for (int j = 0; j < primary_count; j++)
            // {
            //     printf("%d ", n_primary[j]);
            // }
            // printf("\n");

            if (primary_count)
            {
                mine_kernel<<<1,1>>>(alloc, 
                                    n_pattern, 
                                    projection_items, new_item_count, projection_start, projection_end, projection_utility, new_transaction_count,
                                    n_primary, primary_count, local_util_counter,
                                    minUtil, high_utility_patterns);

                // mine_kernel_d(alloc, 
                //                     n_pattern, 
                //                     projection_items, new_item_count, projection_start, projection_end, projection_utility, new_transaction_count,
                //                     n_primary, primary_count, local_util_counter,
                //                     minUtil, high_utility_patterns);
            }
        }

        // printf("\n");
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
        std::map<std::vector<int>, Transaction, VectorComparator> transactions;
        std::vector<uint32_t> primary;

        _read_file(transactions, primary);

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
                items.push_back(Item{static_cast<uint32_t>(t.first[i]), static_cast<uint32_t>(t.second.vals[i])});
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

        mine_kernel<<<1, 1>>>(alloc, d_pattern, d_items, items.size(), d_start, d_end, d_utility, transactions.size(),
                              d_primary, primary.size(), max_item,
                              minUtil, d_high_utility_patterns);


        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();

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
                // high_utility_patterns_str += std::to_string(d_high_utility_patterns[i + 2]) + " ";
                high_utility_patten.push_back(d_high_utility_patterns[i + 2]);
                i++;
            }
            // high_utility_patterns_str += "| ";
            // convert all except the last one
            // if empty, skip
            if (high_utility_patten.size() == 0)
            {
                continue;
            }

            for (int j = 0; j < high_utility_patten.size() - 1; j++)
            {
                high_utility_patterns_str += std::to_string(high_utility_patten[j]) + " ";
            }
            // Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];
            // std::cout << high_utility_patterns_str << ": " << high_utility_patten[high_utility_patten.size() - 1] << std::endl;
            Patterns[high_utility_patterns_str] = high_utility_patten[high_utility_patten.size() - 1];

            high_utility_patterns_str = "";
            high_utility_patten.clear();


        }
        // // std::cout << high_utility_patterns_str << std::endl;
        // for (const auto &p : Patterns)
        // {
        //     std::cout << p.first << " : " << p.second << std::endl;
        // }


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
    void save(const std::string &outFile) {
        std::ofstream writer(outFile);
        if (!writer) {
            std::cerr << "Error opening output file: " << outFile << "\n";
            return;
        }
        // Here we write the discovered patterns (stored in Patterns)
        for (const auto &entry : Patterns) {
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

    void _read_file(std::map<std::vector<int>, Transaction, VectorComparator> &filteredTransactions,
                    std::vector<uint32_t> &primary)
    {
        // File data: each entry is a pair: (vector of item strings, vector of utility ints)
        std::vector<std::pair<std::vector<std::string>, std::vector<int>>> fileData;
        // TWU dictionary: item string -> total weight
        std::unordered_map<std::string, int> twu;

        std::ifstream infile(inputFile);
        if (!infile)
        {
            std::cerr << "Error opening file: " << inputFile << "\n";
            return;
        }
        std::string line;
        while (std::getline(infile, line))
        {
            // Expected format: items : weight : utility_list
            // Items and utility_list are further separated by sep.
            std::vector<std::string> parts = split(line, ":");
            if (parts.size() < 3)
                continue;

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
                filteredTransactions[key] = Transaction{key, val, 0};
            }
            else
            {
                // If the transaction already exists, add corresponding utilities.
                Transaction &trans = filteredTransactions[key];
                for (size_t i = 0; i < val.size(); ++i)
                    trans.vals[i] += val[i];
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
        for (const auto &p : subtree)
        {
            if (p.second >= minUtil)
                primary.push_back(p.first);
        }
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
