#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cstring> // for strcmp
#include <stdexcept>
#include <stdint.h>
#include <numeric>
#include <tuple>

#define KILO 1024ULL
#define MEGA KILO *KILO
#define GIGA KILO *MEGA // 1 GB

// nvcc -arch=sm_86 -rdc=true cuEFIM.cu -o cuEFIM
// ./cuEFIM data.txt \\s 50

struct Item
{
    uint32_t item;
    uint32_t util;
};

// Function to split a string by a delimiter
std::vector<std::string> split(const std::string &str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

auto readFile(std::string filename, char delimiter, uint32_t minUtil)
{
    std::vector<std::vector<Item>> transactions;
    std::map<uint32_t, uint32_t> twu;

    // Open the file.
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        // return 1;
    }

    std::string line;
    // Process the file line by line.
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::vector<std::string> parts = split(line, ':');
        if (parts.size() < 3)
        {
            std::cerr << "Invalid line: " << line << std::endl;
            exit(1);
        }

        // Parse the first part: items (space separated)
        std::vector<std::string> itemTokens = split(parts[0], delimiter);

        // Parse the second part: transaction weight (a single integer)
        uint32_t transactionWeight = 0;
        try
        {
            transactionWeight = std::stoi(parts[1]);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Invalid transaction weight in line: " << line << "\n"
                      << "Exception: " << e.what() << std::endl;
            continue;
        }

        // Parse the third part: utilities (space separated)
        std::vector<std::string> utilTokens = split(parts[2], delimiter);
        if (itemTokens.size() != utilTokens.size())
        {
            std::cerr << "Mismatch between number of items and utilities in line: " << line << std::endl;
            continue;
        }

        // Build the current transaction.
        std::vector<Item> transaction;
        for (size_t i = 0; i < itemTokens.size(); ++i)
        {
            uint32_t itemId = 0;
            uint32_t util = 0;
            try
            {
                itemId = std::stoi(itemTokens[i]);
                util = std::stoi(utilTokens[i]);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing item or utility in line: " << line << "\n"
                          << "Exception: " << e.what() << std::endl;
                continue;
            }

            transaction.push_back({itemId, util});
            // Update TWU for each item by adding the transaction weight.
            twu[itemId] += transactionWeight;
        }
        transactions.push_back(transaction);
    }
    file.close();

    // Filter the TWU map based on minUtil.
    uint32_t maxItem = 0; // max item id in the dataset

    std::map<uint32_t, uint32_t> filtered_twu;
    for (const auto &p : twu)
    {
        if (p.first > maxItem)
            maxItem = p.first;
        if (p.second >= minUtil)
            filtered_twu[p.first] = p.second;
    }

    // Sort the filtered TWU items by utility in descending order.
    std::vector<std::pair<uint32_t, uint32_t>> twuVec(filtered_twu.begin(), filtered_twu.end());
    std::sort(twuVec.begin(), twuVec.end(),
              [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b)
              {
                  return a.second > b.second;
              });

    // sort transactions by twu if they exist in twuVec

    std::vector<uint32_t> subtreeUtility(maxItem + 1, 0);

    for (auto &transaction : transactions)
    {
        // take the transaction.
        // remove items that are not in twu map
        // sort the transaction by twu
        transaction.erase(std::remove_if(transaction.begin(), transaction.end(),
                                         [&filtered_twu](const Item &item)
                                         { return filtered_twu.find(item.item) == filtered_twu.end(); }),
                          transaction.end());
        std::sort(transaction.begin(), transaction.end(),
                  [&filtered_twu](const Item &a, const Item &b)
                  { return filtered_twu.at(a.item) < filtered_twu.at(b.item); });

        auto sum = std::accumulate(transaction.begin(), transaction.end(), 0,
                                   [&filtered_twu](uint32_t acc, const Item &item)
                                   { return acc + item.util; });
        for (const auto &item : transaction)
        {
            subtreeUtility[item.item] += sum;
            sum -= item.util;
        }
    }

    //  return items, start, end, subtreeUtil, localUtil, maxItem

    std::vector<uint32_t> start;
    std::vector<uint32_t> end;
    std::vector<Item> items;

    auto curr = 0;
    for (size_t i = 0; i < transactions.size(); ++i)
    {
        start.push_back(curr);
        for (size_t j = 0; j < transactions[i].size(); ++j)
        {
            items.push_back(transactions[i][j]);
            curr++;
        }
        end.push_back(curr);
    }

    return std::make_tuple(items, start, end, subtreeUtility, filtered_twu, maxItem);
}

// mine<<<1, 1>>>(base_pattern, d_items, items.size(), d_start, d_end, d_utility, start.size(), d_primary, d_secondary, maxItem, d_pattern);

__device__ void d_mine(uint32_t *base_pattern,
                       Item *items, uint32_t nItems,
                       uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
                       uint32_t *primary, uint32_t *secondary, uint32_t maxItem,
                       uint32_t *pattern, uint32_t minUtil, uint32_t *that_didnt_work)

{
    for (uint32_t i = 0; i < maxItem + 1; i++)
    {
        if (primary[i] == 0)
        {
            continue;
        }

        uint32_t *n_pattern = (uint32_t *)malloc(sizeof(uint32_t) * (base_pattern[0] + 1));
        n_pattern[0] = base_pattern[0] + 1;
        for (uint32_t j = 0; j < base_pattern[0]; j++)
        {
            n_pattern[j + 1] = base_pattern[j + 1];
        }
        n_pattern[n_pattern[0]] = i;

        // create projection, local, subtree, itemcount...
        Item *projection_items = (Item *)malloc(sizeof(Item) * nItems);
        uint32_t *n_start = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);
        uint32_t *n_end = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);
        uint32_t *n_utility = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);

        uint32_t *n_subtree = (uint32_t *)malloc(sizeof(uint32_t) * (maxItem + 1));
        uint32_t *n_local = (uint32_t *)malloc(sizeof(uint32_t) * (maxItem + 1));

        uint32_t itemCounter = 0;
        uint32_t nTransactionsCounter = 0;

        uint32_t patternUtil = 0;

        for (uint32_t j = 0; j < nTransactions; j++)
        {
            bool flag = false;
            uint32_t temp_util = 0;
            uint32_t temp_start = 0;
            for (uint32_t k = start[j]; k < end[j]; k++)
            {

                if (flag)
                {
                    // check if item is in secondary and add to projection
                    if (secondary[items[k].item])
                    {
                        projection_items[itemCounter] = items[k];
                        temp_util += items[k].util;
                        itemCounter++;
                    }
                }

                if (items[k].item == i)
                {
                    flag = true;
                    n_start[nTransactionsCounter] = itemCounter;
                    n_utility[nTransactionsCounter] = items[k].util + utility[j];
                    patternUtil += n_utility[nTransactionsCounter];
                    temp_util = n_utility[nTransactionsCounter];
                    temp_start = k + 1;
                }
            }
            if (flag)
            {
                if (itemCounter != n_start[nTransactionsCounter])
                {
                    n_end[nTransactionsCounter] = itemCounter;
                    nTransactionsCounter++;
                }
                uint32_t temp = 0;
                for (uint32_t k = temp_start; k < end[j]; k++)
                {
                    // check if item is in secondary and add to projection
                    if (secondary[items[k].item])
                    {
                        n_subtree[items[k].item] += temp_util - temp;
                        n_local[items[k].item] += temp_util;
                        temp += items[k].util;
                    }
                }
            }
        }

        if (patternUtil >= minUtil)
        {

            // printf("Pattern: ");
            // for (uint32_t j = 0; j < n_pattern[0]; j++)
            // {
            //     printf("%d ", n_pattern[j + 1]);
            // }
            // printf(": %d\n", patternUtil);


            atomicAdd(&that_didnt_work[0], 1);
        }

        if (nTransactionsCounter == 0)
        {
            free(projection_items);
            free(n_start);
            free(n_end);
            free(n_utility);
            free(n_subtree);
            free(n_local);
            continue;
        }

        uint32_t primary_count = 0;
        for (uint32_t j = 0; j < maxItem; j++)
        {
            if (n_subtree[j] >= minUtil)
            {
                n_subtree[j] = j;
                primary_count++;
            }
            else
            {
                n_subtree[j] = 0;
            }

            if (n_local[j] >= minUtil)
            {
                n_local[j] = j;
            }
            else
            {
                n_local[j] = 0;
            }
        }

        if (primary_count == 0)
        {
            free(projection_items);
            free(n_start);
            free(n_end);
            free(n_utility);
            free(n_subtree);
            free(n_local);
            continue;
        }
        else
        {
            d_mine(n_pattern, projection_items, itemCounter, n_start, n_end, n_utility, nTransactionsCounter, n_subtree, n_local, maxItem, pattern, minUtil, that_didnt_work);
        }
    }
    // free all the memory
    free(base_pattern);
    free(items);
    free(start);
    free(end);
    free(utility);
    free(primary);
    free(secondary);
}

// __global__ void mine(uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t *primary, uint32_t *secondary, Item *items, uint32_t nTransactions, uint32_t nItems, uint32_t *pattern, uint32_t maxItem)
__global__ void mine(uint32_t *base_pattern,
                     Item *items, uint32_t nItems,
                     uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
                     uint32_t *primary, uint32_t *secondary, uint32_t maxItem,
                     uint32_t *pattern, uint32_t minUtil, uint32_t *that_didnt_work)

{
    // for (uint32_t i = 0; i < maxItem + 1; i++)
    // {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid > maxItem + 1) return;
        uint32_t i = tid;
            if (primary[i] == 0)
            {
                return;
            }
        printf("i: %d\n", i);


        uint32_t *n_pattern = (uint32_t *)malloc(sizeof(uint32_t) * (base_pattern[0] + 1));
        n_pattern[0] = base_pattern[0] + 1;
        for (uint32_t j = 0; j < base_pattern[0]; j++)
        {
            n_pattern[j + 1] = base_pattern[j + 1];
        }
        n_pattern[n_pattern[0]] = i;

        // create projection, local, subtree, itemcount...
        Item *projection_items = (Item *)malloc(sizeof(Item) * nItems);
        uint32_t *n_start = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);
        uint32_t *n_end = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);
        uint32_t *n_utility = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);

        uint32_t *n_subtree = (uint32_t *)malloc(sizeof(uint32_t) * (maxItem + 1));
        uint32_t *n_local = (uint32_t *)malloc(sizeof(uint32_t) * (maxItem + 1));

        // check if memory is allocated
        if (projection_items == NULL || n_start == NULL || n_end == NULL || n_utility == NULL || n_subtree == NULL || n_local == NULL)
        {
            printf("Memory allocation failed\n");
            return;
        }

        uint32_t itemCounter = 0;
        uint32_t nTransactionsCounter = 0;

        uint32_t patternUtil = 0;

        for (uint32_t j = 0; j < nTransactions; j++)
        {
            bool flag = false;
            uint32_t temp_util = 0;
            uint32_t temp_start = 0;
            for (uint32_t k = start[j]; k < end[j]; k++)
            {

                if (flag)
                {
                    // check if item is in secondary and add to projection
                    if (secondary[items[k].item])
                    {
                        projection_items[itemCounter] = items[k];
                        temp_util += items[k].util;
                        itemCounter++;
                    }
                }

                if (items[k].item == i)
                {
                    flag = true;
                    n_start[nTransactionsCounter] = itemCounter;
                    n_utility[nTransactionsCounter] = items[k].util + utility[j];
                    patternUtil += n_utility[nTransactionsCounter];
                    temp_util = n_utility[nTransactionsCounter];
                    temp_start = k + 1;
                }
            }
            if (flag)
            {
                if (itemCounter != n_start[nTransactionsCounter])
                {
                    n_end[nTransactionsCounter] = itemCounter;
                    nTransactionsCounter++;
                }
                uint32_t temp = 0;
                for (uint32_t k = temp_start; k < end[j]; k++)
                {
                    // check if item is in secondary and add to projection
                    if (secondary[items[k].item])
                    {
                        n_subtree[items[k].item] += temp_util - temp;
                        n_local[items[k].item] += temp_util;
                        temp += items[k].util;
                    }
                }
            }
        }
        printf("util: %d\n", patternUtil);

        if (patternUtil >= minUtil)
        {

            // printf("Pattern: ");
            // for (uint32_t j = 0; j < n_pattern[0]; j++)
            // {
            //     printf("%d ", n_pattern[j + 1]);
            // }
            // printf(": %d\n", patternUtil);


            atomicAdd(&that_didnt_work[0], 1);

        }

        if (nTransactionsCounter == 0)
        {
            free(projection_items);
            free(n_start);
            free(n_end);
            free(n_utility);
            free(n_subtree);
            free(n_local);
            return;
        }

        uint32_t primary_count = 0;
        for (uint32_t j = 0; j < maxItem; j++)
        {
            if (n_subtree[j] >= minUtil)
            {
                n_subtree[j] = j;
                primary_count++;
            }
            else
            {
                n_subtree[j] = 0;
            }

            if (n_local[j] >= minUtil)
            {
                n_local[j] = j;
            }
            else
            {
                n_local[j] = 0;
            }
        }

        if (primary_count == 0)
        {
            free(projection_items);
            free(n_start);
            free(n_end);
            free(n_utility);
            free(n_subtree);
            free(n_local);
            return;
        }
        else
        {
            mine<<<maxItem+1, 1>>>(n_pattern, projection_items, itemCounter, n_start, n_end, n_utility, nTransactionsCounter, n_subtree, n_local, maxItem, pattern, minUtil, that_didnt_work);
            // d_mine(n_pattern, projection_items, itemCounter, n_start, n_end, n_utility, nTransactionsCounter, n_subtree, n_local, maxItem, pattern, minUtil, that_didnt_work);
        }
    // }
    // free all the memory
    // free(base_pattern);
    // free(items);
    // free(start);
    // free(end);
    // free(utility);
    // free(primary);
    // free(secondary);
}


__global__ void print_pattern(uint32_t *pattern)
{
    for (uint32_t i = 0; i < pattern[0]; i++)
    {
        printf("%d ", pattern[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{

    // set cuda heap size to 4GB
    ssize_t heapSize = 6 * GIGA;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);

    // Parse command-line arguments:
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <delimiter> <minUtil>" << std::endl;
        return 1;
    }

    std::string filename(argv[1]);
    char delimiter;
    uint32_t minUtil;

    try
    {
        // Allow special arguments for space and tab delimiters.
        if (strcmp(argv[2], "\\s") == 0)
        {
            delimiter = ' ';
        }
        else if (strcmp(argv[2], "\\t") == 0)
        {
            delimiter = '\t';
        }
        else
        {
            delimiter = argv[2][0];
        }

        minUtil = std::stoi(argv[3]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    // read File
    auto [items, start, end, subtreeUtil, localUtil, maxItem] = readFile(filename, delimiter, minUtil);
    std::cout << "Read file" << std::endl;

    std::vector<uint32_t> primary;
    for (size_t i = 0; i < maxItem + 1; ++i)
    {
        if (subtreeUtil[i] >= minUtil)
        {
            primary.push_back(i);
        }
        else
        {
            primary.push_back(0);
        }
    }

    std::vector<uint32_t> secondary(maxItem + 1, 0);
    for (size_t i = 0; i < maxItem + 1; ++i)
    {
        if (localUtil[i] >= minUtil)
        {
            secondary[i] = i;
        }
        else
        {
            secondary[i] = 0;
        }
    }

    // // print db
    // for (size_t i = 0; i < start.size(); ++i)
    // {
    //     for (size_t j = start[i]; j < end[i]; ++j)
    //     {
    //         std::cout << items[j].item << ":" << items[j].util << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // print primary
    // std::cout << "Primary: ";
    // for (size_t i = 0; i < primary.size(); ++i)
    // {
    //     std::cout << primary[i] << " ";
    // }
    // std::cout << std::endl;

    // // print secondary
    // std::cout << "Secondary: ";
    // for (size_t i = 0; i < secondary.size(); ++i)
    // {
    //     std::cout << secondary[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;

    // allocate all of them on cuda
    uint32_t *d_start, *d_end, *d_utility, *d_primary, *d_secondary;
    Item *d_items;

    cudaMalloc(&d_start, start.size() * sizeof(uint32_t));
    cudaMalloc(&d_end, end.size() * sizeof(uint32_t));
    cudaMalloc(&d_primary, primary.size() * sizeof(uint32_t));
    cudaMalloc(&d_secondary, secondary.size() * sizeof(uint32_t));
    cudaMalloc(&d_items, items.size() * sizeof(Item));
    cudaMalloc(&d_utility, start.size() * sizeof(uint32_t));

    cudaMemcpy(d_start, start.data(), start.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end.data(), end.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_secondary, secondary.data(), secondary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_items, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice);
    cudaMemset(d_utility, 0, start.size() * sizeof(uint32_t));

    // allocate pattern holder
    uint32_t *d_pattern;
    cudaMalloc(&d_pattern, MEGA); // 1MB for pattern

    uint32_t *base_pattern;
    cudaMalloc(&base_pattern, 1 * sizeof(uint32_t));
    cudaMemset(base_pattern, 0, 1 * sizeof(uint32_t));

    uint32_t *that_didnt_work;
    cudaMallocManaged(&that_didnt_work, 1 * sizeof(uint32_t));
    cudaMemset(that_didnt_work, 0, 1 * sizeof(uint32_t));

    // call kernel
    std::cout << "Calling kernel" << std::endl;

    // block size is 1, grid size is 

    mine<<<maxItem + 1, 1>>>(base_pattern, d_items, items.size(), d_start, d_end, d_utility, start.size(), d_primary, d_secondary, maxItem, d_pattern, minUtil, that_didnt_work);
    cudaDeviceSynchronize();

    print_pattern<<<1, 1>>>(d_pattern);
    cudaDeviceSynchronize();

    // copy pattern to host
    std::vector<uint32_t> pattern(1 * sizeof(uint32_t));
    cudaMemcpy(pattern.data(), base_pattern, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // // print pattern until what the 0 says
    // std::cout << "Pattern[0]: " << pattern[0] << std::endl;
    // for (size_t i = 0; i < pattern[0]; i++)
    // {
    //     if (pattern[i] == 0)
    //     {
    //         break;
    //     }
    //     std::cout << "Pattern: ";
    //     for (size_t j = 0; j < pattern[i]; j++)
    //     {
    //         std::cout << pattern[i + j + 1] << " ";
    //     }
    //     std::cout << ": " << pattern[i + pattern[i] + 1] << std::endl;
    //     i += pattern[i] + 2;
    // }

    // print that didnt work
    std::cout << "That didn't work: " << that_didnt_work[0] << std::endl;

    return 0;
}