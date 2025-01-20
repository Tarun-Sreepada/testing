#include "mine.cuh"
#include <cstdio>
#include <cstdlib>


//
// The main mining kernel launched from the host.
// It uses dynamic parallelism (launching child kernels) to continue the recursive mining.
//
__global__ void mine(BumpAllocator *alloc, uint32_t *base_pattern,
                     Item *items, uint32_t nItems,
                     uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
                     uint32_t *primary, uint32_t *secondary, uint32_t maxItem,
                     uint32_t minUtil, uint32_t *pattern_counter)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > maxItem + 1) return;
    uint32_t i = tid;
    if (primary[i] == 0)
        return;

    // uint32_t *n_pattern = (uint32_t *)malloc(sizeof(uint32_t) * (base_pattern[0] + 1));
    uint32_t *n_pattern = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * (base_pattern[0] + 1));
    if (n_pattern == nullptr) return;

    n_pattern[0] = base_pattern[0] + 1;
    for (uint32_t j = 0; j < base_pattern[0]; j++)
    {
        n_pattern[j + 1] = base_pattern[j + 1];
    }
    n_pattern[n_pattern[0]] = i;

    // Item *projection_items = (Item *)malloc(sizeof(Item) * nItems);
    // uint32_t *n_start = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);
    // uint32_t *n_end = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);
    // uint32_t *n_utility = (uint32_t *)malloc(sizeof(uint32_t) * nTransactions);
    Item *projection_items = (Item *)bump_alloc(alloc, sizeof(Item) * nItems);
    uint32_t *n_start = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * nTransactions);
    uint32_t *n_end = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * nTransactions);
    uint32_t *n_utility = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * nTransactions);

    // uint32_t *n_subtree = (uint32_t *)malloc(sizeof(uint32_t) * (maxItem + 1));
    // uint32_t *n_local = (uint32_t *)malloc(sizeof(uint32_t) * (maxItem + 1));
    uint32_t *n_subtree = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * (maxItem + 1));
    uint32_t *n_local = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * (maxItem + 1));
    for (uint32_t j = 0; j < maxItem + 1; j++) {
        n_subtree[j] = 0;
        n_local[j] = 0;
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
        atomicAdd(&pattern_counter[0], 1);
    }

    // if (nTransactionsCounter == 0)
    // {
    //     free(projection_items);
    //     free(n_start);
    //     free(n_end);
    //     free(n_utility);
    //     free(n_subtree);
    //     free(n_local);
    //     free(n_pattern);
    //     return;
    // }

    uint32_t primary_count = 0;
    for (uint32_t j = 0; j < maxItem; j++)
    {
        if (n_subtree[j] >= minUtil)
        {
            n_subtree[j] = j;
            primary_count++;
        }
        else
            n_subtree[j] = 0;

        if (n_local[j] >= minUtil)
            n_local[j] = j;
        else
            n_local[j] = 0;
    }

    if (primary_count == 0)
    {
    //     free(projection_items);
    //     free(n_start);
    //     free(n_end);
    //     free(n_utility);
    //     free(n_subtree);
    //     free(n_local);
    //     free(n_pattern);
    //     return;
    }
    else
    {
        // Launch a child kernel using dynamic parallelism.
        mine<<<maxItem+1, 1>>>(alloc, n_pattern, projection_items, itemCounter, n_start, n_end,
                                 n_utility, nTransactionsCounter, n_subtree, n_local,
                                 maxItem, minUtil, pattern_counter);
    }

    // free(n_pattern);
    // free(projection_items);
    // free(n_start);
    // free(n_end);
    // free(n_utility);
    // free(n_subtree);
    // free(n_local);
}

//
// A helper kernel to print the pattern stored in device memory.
//
__global__ void print_pattern(uint32_t *pattern)
{
    for (uint32_t i = 0; i < pattern[0]; i++)
    {
        printf("%d ", pattern[i]);
    }
    printf("\n");
}
