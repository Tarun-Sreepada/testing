#include "mine.cuh"
#include <cstdio>
#include <cstdlib>

//
// The main mining kernel launched from the host.
// It uses dynamic parallelism (launching child kernels) to continue the recursive mining.
//
__device__ void mine_d(BumpAllocator *alloc, uint32_t *base_pattern,
                     Item *items, uint32_t nItems,
                     uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
                     uint32_t *primary, uint32_t maxItem,
                     uint32_t minUtil, uint32_t *pattern_counter)
{

    uint32_t scratch_pad_total = sizeof(Item) * nItems + 
                                 sizeof(uint32_t) * nTransactions * 3 + 
                                 sizeof(uint32_t) * (maxItem + 1);
    
    void *scratch_pad = bump_alloc(alloc, scratch_pad_total);

    Item *scratch_items = (Item *)scratch_pad;
    scratch_pad += sizeof(Item) * nItems;
    uint32_t *scratch_start = (uint32_t *)scratch_pad;
    scratch_pad += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_end = (uint32_t *)scratch_pad;
    scratch_pad += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_utility = (uint32_t *)scratch_pad;
    scratch_pad += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_local = (uint32_t *)scratch_pad;


    for (int i = 0; i < maxItem + 1; i++)
    {
        if (primary[i] == 0)
            continue;

        // reset local utility
        for (uint32_t j = 0; j < maxItem + 1; j++)
        {
            scratch_local[j] = 0;
        }

        // reset start and 

        uint32_t *n_pattern = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * (base_pattern[0] + 2));

        n_pattern[0] = base_pattern[0] + 1;
        for (uint32_t j = 0; j < base_pattern[0]; j++)
        {
            n_pattern[j + 1] = base_pattern[j + 1];
        }
        n_pattern[n_pattern[0]] = i;


        // printf("Pattern: ");
        // for (uint32_t j = 1; j < n_pattern[0] +1;j++)
        // {
        //     printf("%d ", n_pattern[j]);
        // }
        // printf("\n");

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
                    scratch_items[itemCounter] = items[k];
                    temp_util += items[k].util;
                    // printf("%d:%d ", items[k].item, items[k].util);
                    itemCounter++;
                }
                if (items[k].item == i)
                {
                    flag = true;
                    scratch_start[nTransactionsCounter] = itemCounter;
                    scratch_utility[nTransactionsCounter] = items[k].util + utility[j];
                    patternUtil += scratch_utility[nTransactionsCounter];
                    temp_util = scratch_utility[nTransactionsCounter];
                    temp_start = k + 1;
                }
            }
            if (flag)
            {
                // printf("||%d||\n", scratch_utility[nTransactionsCounter]);
                if (itemCounter != scratch_start[nTransactionsCounter])
                {
                    scratch_end[nTransactionsCounter] = itemCounter;
                    nTransactionsCounter++;
                }
                uint32_t temp = 0;
                for (uint32_t k = temp_start; k < end[j]; k++)
                {
                    scratch_local[items[k].item] += temp_util;
                    temp += items[k].util;
                }
            }
        }

        // printf("Num new transactions: %d\n", nTransactionsCounter);
        // // print start and ends
        // for (uint32_t j = 0; j < nTransactionsCounter; j++)
        // {
        //     printf("Start: %d, End: %d\n", scratch_start[j], scratch_end[j]);
        // }

        // printf("Local Utility: ");
        // for (uint32_t j = 0; j < maxItem + 1; j++)
        // {
        //     printf("%d:%d ", j, scratch_local[j]);
        // }
        // printf("\n");

        if (patternUtil >= minUtil){
            atomicAdd(&pattern_counter[0], 1);
            // print pattern and util
            // for (uint32_t j = 1; j < n_pattern[0] + 1; j++)
            // {
            //     printf("%d ", n_pattern[j]);
            // }
            // printf("||%d\n", patternUtil);
            
        }


        // allocate new database if we have transactions;
        if (nTransactionsCounter)
        {

            uint32_t total_bytes_req = sizeof(Item) * itemCounter + 
                                        sizeof(uint32_t) * nTransactionsCounter * 3 + 
                                        sizeof(uint32_t) * (maxItem + 1);

            void *ptr = bump_alloc(alloc, total_bytes_req);

            Item *projection_items = (Item *)ptr;
            ptr += sizeof(Item) * itemCounter;
            uint32_t *n_start = (uint32_t *)ptr;
            ptr += sizeof(uint32_t) * nTransactionsCounter;
            uint32_t *n_end = (uint32_t *)ptr;
            ptr += sizeof(uint32_t) * nTransactionsCounter;
            uint32_t *n_utility = (uint32_t *)ptr;
            ptr += sizeof(uint32_t) * nTransactionsCounter;
            uint32_t *n_subtree = (uint32_t *)ptr;
            

            uint32_t new_item_count = 0;
            uint32_t new_transaction_count = 0;

            // copy over the items if they have local utility greater than minUtil
            for (uint32_t j = 0; j < nTransactions; j++)
            {
                // check if start and end are the same
                // printf("Start: %d, End: %d\n", scratch_start[j], scratch_end[j]);
                uint32_t temp_util = scratch_utility[j];
                if (scratch_start[j] == scratch_end[j])
                    continue;

                n_start[new_transaction_count] = new_item_count;
                for (uint32_t k = scratch_start[j]; k < scratch_end[j]; k++)
                {
                    if (scratch_local[scratch_items[k].item] >= minUtil)
                    {
                        projection_items[new_item_count] = scratch_items[k];
                        temp_util += scratch_items[k].util;
                        new_item_count++;
                    }
                }
                if (new_item_count != scratch_start[j])
                {
                    n_end[new_transaction_count] = new_item_count;
                    n_utility[new_transaction_count] = scratch_utility[j];
                    new_transaction_count++;
                }

                uint32_t temp = 0;
                for (uint32_t k = n_start[new_transaction_count - 1]; k < n_end[new_transaction_count - 1]; k++)
                {
                    n_subtree[projection_items[k].item] += temp_util - temp;
                    temp += projection_items[k].util;
                }

                if (new_transaction_count == nTransactionsCounter)
                    break;
            }


            // // print new transactions
            // printf("New Transactions: \n");
            // for (uint32_t j = 0; j < new_transaction_count; j++)
            // {
            //     printf("%d||", n_utility[j]);
            //     for (uint32_t k = n_start[j]; k < n_end[j]; k++)
            //     {
            //         printf("%d:%d ", projection_items[k].item, projection_items[k].util);
            //     }
            //     printf("\n");
            // }

            // // print new subtree
            // printf("New Subtree: ");
            // for (uint32_t j = 0; j < maxItem + 1; j++)
            // {
            //     printf("%d:%d ", j, n_subtree[j]);
            // }
            // printf("\n");


            // clean up subtree
            uint32_t primary_count = 0;
            for (uint32_t j = 0; j < maxItem + 1; j++)
            {
                if (n_subtree[j] >= minUtil)
                {
                    n_subtree[j] = j;
                    primary_count++;
                }
                else
                    n_subtree[j] = 0;
            }

            
            // printf("Primary Count: %d\n", primary_count);
            // printf("Pattern: ");
            // for (uint32_t j = 0; j < n_pattern[0] +1;j++)
            // {
            //     printf("%d ", n_pattern[j]);
            // }
            // printf("\n");


            // printf("\n");
            // call mine recursively
            if (primary_count)
            {
                // mine<<<1, 1>>>(alloc, n_pattern, projection_items, new_item_count, n_start, n_end,
                //                n_utility, new_transaction_count, n_subtree,
                //                maxItem, minUtil, pattern_counter);
                mine_d(alloc, n_pattern, projection_items, new_item_count, n_start, n_end,
                               n_utility, new_transaction_count, n_subtree,
                               maxItem, minUtil, pattern_counter);
            }
        }

        // printf("=========================\n\n\n");
    }
}


__global__ void mine(BumpAllocator *alloc, uint32_t *base_pattern,
                     Item *items, uint32_t nItems,
                     uint32_t *start, uint32_t *end, uint32_t *utility, uint32_t nTransactions,
                     uint32_t *primary, uint32_t numPrimary, uint32_t maxItem,
                     uint32_t minUtil, uint32_t *pattern_counter)
{

    uint32_t scratch_pad_total = sizeof(Item) * nItems + 
                                 sizeof(uint32_t) * nTransactions * 3 + 
                                 sizeof(Item) * (maxItem + 1);
    
    void *scratch_pad = bump_alloc(alloc, scratch_pad_total);

    Item *scratch_items = (Item *)scratch_pad;
    scratch_pad += sizeof(Item) * nItems;
    uint32_t *scratch_start = (uint32_t *)scratch_pad;
    scratch_pad += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_end = (uint32_t *)scratch_pad;
    scratch_pad += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_utility = (uint32_t *)scratch_pad;
    scratch_pad += sizeof(uint32_t) * nTransactions;
    uint32_t *scratch_local = (uint32_t *)scratch_pad;


    for (int i = 0; i < numPrimary; i++)
    {
        printf("Primary: %d\n", primary[i]);

        // reset local utility
        for (uint32_t j = 0; j < maxItem + 1; j++)
        {
            scratch_local[j] = 0;
        }

        // reset start and 

        uint32_t *n_pattern = (uint32_t *)bump_alloc(alloc, sizeof(uint32_t) * (base_pattern[0] + 2));

        n_pattern[0] = base_pattern[0] + 1;
        for (uint32_t j = 0; j < base_pattern[0]; j++)
        {
            n_pattern[j + 1] = base_pattern[j + 1];
        }
        n_pattern[n_pattern[0]] = i;


        // printf("Pattern: ");
        // for (uint32_t j = 1; j < n_pattern[0] +1;j++)
        // {
        //     printf("%d ", n_pattern[j]);
        // }
        // printf("\n");

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
                    scratch_items[itemCounter] = items[k];
                    temp_util += items[k].util;
                    // printf("%d:%d ", items[k].item, items[k].util);
                    itemCounter++;
                }
                if (items[k].item == i)
                {
                    flag = true;
                    scratch_start[nTransactionsCounter] = itemCounter;
                    scratch_utility[nTransactionsCounter] = items[k].util + utility[j];
                    patternUtil += scratch_utility[nTransactionsCounter];
                    temp_util = scratch_utility[nTransactionsCounter];
                    temp_start = k + 1;
                }
            }
            if (flag)
            {
                // printf("||%d||\n", scratch_utility[nTransactionsCounter]);
                if (itemCounter != scratch_start[nTransactionsCounter])
                {
                    scratch_end[nTransactionsCounter] = itemCounter;
                    nTransactionsCounter++;
                }
                uint32_t temp = 0;
                for (uint32_t k = temp_start; k < end[j]; k++)
                {
                    scratch_local[items[k].item] += temp_util;
                    temp += items[k].util;
                }
            }
        }

        // printf("Num new transactions: %d\n", nTransactionsCounter);
        // // print start and ends
        // for (uint32_t j = 0; j < nTransactionsCounter; j++)
        // {
        //     printf("Start: %d, End: %d\n", scratch_start[j], scratch_end[j]);
        // }

        // printf("Local Utility: ");
        // for (uint32_t j = 0; j < maxItem + 1; j++)
        // {
        //     printf("%d:%d ", j, scratch_local[j]);
        // }
        // printf("\n");

        if (patternUtil >= minUtil){
            atomicAdd(&pattern_counter[0], 1);
            // print pattern and util
            // for (uint32_t j = 1; j < n_pattern[0] + 1; j++)
            // {
            //     printf("%d ", n_pattern[j]);
            // }
            // printf("||%d\n", patternUtil);
            
        }


        // allocate new database if we have transactions;
        if (nTransactionsCounter)
        {

            uint32_t total_bytes_req = sizeof(Item) * itemCounter + 
                                        sizeof(uint32_t) * nTransactionsCounter * 3 + 
                                        sizeof(uint32_t) * (maxItem + 1);

            void *ptr = bump_alloc(alloc, total_bytes_req);

            Item *projection_items = (Item *)ptr;
            ptr += sizeof(Item) * itemCounter;
            uint32_t *n_start = (uint32_t *)ptr;
            ptr += sizeof(uint32_t) * nTransactionsCounter;
            uint32_t *n_end = (uint32_t *)ptr;
            ptr += sizeof(uint32_t) * nTransactionsCounter;
            uint32_t *n_utility = (uint32_t *)ptr;
            ptr += sizeof(uint32_t) * nTransactionsCounter;
            uint32_t *n_subtree = (uint32_t *)ptr;
            

            uint32_t new_item_count = 0;
            uint32_t new_transaction_count = 0;

            // copy over the items if they have local utility greater than minUtil
            for (uint32_t j = 0; j < nTransactions; j++)
            {
                // check if start and end are the same
                // printf("Start: %d, End: %d\n", scratch_start[j], scratch_end[j]);
                uint32_t temp_util = scratch_utility[j];
                if (scratch_start[j] == scratch_end[j])
                    continue;

                n_start[new_transaction_count] = new_item_count;
                for (uint32_t k = scratch_start[j]; k < scratch_end[j]; k++)
                {
                    if (scratch_local[scratch_items[k].item] >= minUtil)
                    {
                        projection_items[new_item_count] = scratch_items[k];
                        temp_util += scratch_items[k].util;
                        new_item_count++;
                    }
                }
                if (new_item_count != scratch_start[j])
                {
                    n_end[new_transaction_count] = new_item_count;
                    n_utility[new_transaction_count] = scratch_utility[j];
                    new_transaction_count++;
                }

                uint32_t temp = 0;
                for (uint32_t k = n_start[new_transaction_count - 1]; k < n_end[new_transaction_count - 1]; k++)
                {
                    n_subtree[projection_items[k].item] += temp_util - temp;
                    temp += projection_items[k].util;
                }

                if (new_transaction_count == nTransactionsCounter)
                    break;
            }


            // // print new transactions
            // printf("New Transactions: \n");
            // for (uint32_t j = 0; j < new_transaction_count; j++)
            // {
            //     printf("%d||", n_utility[j]);
            //     for (uint32_t k = n_start[j]; k < n_end[j]; k++)
            //     {
            //         printf("%d:%d ", projection_items[k].item, projection_items[k].util);
            //     }
            //     printf("\n");
            // }

            // // print new subtree
            // printf("New Subtree: ");
            // for (uint32_t j = 0; j < maxItem + 1; j++)
            // {
            //     printf("%d:%d ", j, n_subtree[j]);
            // }
            // printf("\n");


            // clean up subtree
            uint32_t primary_count = 0;
            for (uint32_t j = 0; j < maxItem + 1; j++)
            {
                if (n_subtree[j] >= minUtil)
                {
                    n_subtree[j] = j;
                    primary_count++;
                }
                else
                    n_subtree[j] = 0;
            }

            
            // printf("Primary Count: %d\n", primary_count);
            // printf("Pattern: ");
            // for (uint32_t j = 0; j < n_pattern[0] +1;j++)
            // {
            //     printf("%d ", n_pattern[j]);
            // }
            // printf("\n");


            // printf("\n");
            // call mine recursively
            if (primary_count)
            {
                // mine<<<1, 1>>>(alloc, n_pattern, projection_items, new_item_count, n_start, n_end,
                //                n_utility, new_transaction_count, n_subtree,
                //                maxItem, minUtil, pattern_counter);
                // mine_d(alloc, n_pattern, projection_items, new_item_count, n_start, n_end,
                //                n_utility, new_transaction_count, n_subtree,
                //                maxItem, minUtil, pattern_counter);
            }
        }

        // printf("=========================\n\n\n");
    }
}
