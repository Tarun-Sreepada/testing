// #pragma once
// #include <cstdint>
// #include "database.cuh"
// #include "memory.cuh"
// #include "work.cuh"
// #include <cstring>

// // Binary Search Utility
// __device__ int binarySearchItems(const Item *items, int n, uint32_t search_id, int offset, int length)
// {
//     if (offset < 0 || offset >= n || length <= 0 || (offset + length) > n)
//         return -1;
//     int l = offset, r = offset + length - 1;
//     while (l <= r)
//     {
//         int mid = l + (r - l) / 2;
//         if (items[mid].key == search_id)
//             return mid;
//         items[mid].key < search_id ? l = mid + 1 : r = mid - 1;
//     }
//     return -1;
// }

// __device__ uint32_t pcg_hash(uint32_t input)
// {
//     uint32_t state = input * 747796405u + 2891336453u;
//     uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
//     return (word >> 22u) ^ word;
// }

// __device__ uint32_t hashFunction(uint32_t key, uint32_t tableSize)
// {
//     return pcg_hash(key) % tableSize;
// }

// __device__ uint32_t items_hasher(const Item *items, int n, int tableSize)
// {
//     uint32_t hash = 0;
//     for (int i = 0; i < n; i++)
//         hash ^= pcg_hash(items[i].key);
//     return hash % tableSize;
// }

// __global__ void copy(
//     CudaMemoryManager *memory_manager,
//     AtomicWorkStack *work_queue,
//     Item *items,
//     int *start,
//     int *end,
//     int *primary,
//     int num_primary,
//     int num_transactions,
//     int max_item)
// {

//     stack_init(work_queue);

//     int item_count = end[num_transactions - 1] - start[0];

//     int bytes_to_allocate = 4 * sizeof(int) +                // pattern
//                             item_count * sizeof(Item) +      // items
//                             num_transactions * sizeof(int) + // start
//                             num_transactions * sizeof(int) + // end
//                             num_transactions * sizeof(int) +   // utility
//                             2 * sizeof(int);  // work_done

//     // void *base_ptr = memory_manager->malloc(bytes_to_allocate);
//     void *base_ptr = deviceMemMalloc(memory_manager, bytes_to_allocate);
//     memset(base_ptr, 0, bytes_to_allocate);

//     void *ptr = base_ptr;
//     int *pattern = reinterpret_cast<int *>(ptr);
//     ptr += 4 * sizeof(int);
//     Item *n_items = reinterpret_cast<Item *>(ptr);
//     ptr += item_count * sizeof(Item);
//     int *n_start = reinterpret_cast<int *>(ptr);
//     ptr += num_transactions * sizeof(int);
//     int *n_end = reinterpret_cast<int *>(ptr);
//     ptr += num_transactions * sizeof(int);
//     int *n_utility = reinterpret_cast<int *>(ptr);
//     ptr += num_transactions * sizeof(int);
//     int *work_done = reinterpret_cast<int *>(ptr);
//     ptr += sizeof(int);

//     pattern[0] = 0;
//     work_done[0] = 0;
//     memcpy(n_start, start, num_transactions * sizeof(int));
//     memcpy(n_end, end, num_transactions * sizeof(int));
//     memcpy(n_items, items, item_count * sizeof(Item));

//     for (int i = 0; i < num_primary; i++)
//     {
//         WorkItem work_item;
//         work_item.pattern = pattern;
//         work_item.items = n_items;
//         work_item.num_items = item_count;
//         work_item.start = n_start;
//         work_item.end = n_end;
//         work_item.utility = n_utility;
//         work_item.num_transactions = num_transactions;
//         work_item.primary_item = primary[i];
//         work_item.max_item = max_item;
//         work_item.work_done = work_done;
//         work_item.work_count = num_primary;
//         work_item.bytes = bytes_to_allocate;
//         work_item.base_ptr = base_ptr;

//         stack_push(work_queue, work_item);
//     }
//     atomicAdd(&work_queue->active, num_primary);
// }

// __global__ void slowdowner(int *slowdown)
// {
//     atomicAdd(slowdown, 1);
// }

// __device__ ProjectionMemory allocateProjectionMemory(CudaMemoryManager *memory_manager,const WorkItem *work_item) {
//     ProjectionMemory pm;
//     pm.bytes_to_alloc =
//           (work_item->pattern[0] + 2) * sizeof(int)         +  // pattern
//           work_item->num_items * sizeof(Item)               +  // items
//           work_item->num_transactions * sizeof(int)         +  // start array
//           work_item->num_transactions * sizeof(int)         +  // end array
//           work_item->num_transactions * sizeof(int)         +  // utility per transaction
//           work_item->num_transactions * sizeof(Item) * scale +  // tran_hash (hash table)
//           work_item->max_item * sizeof(Item) * scale         +  // local_util
//           work_item->max_item * sizeof(Item) * scale         +  // subtree_util
//           1 * sizeof(int);                                     // work_done

//     // pm.base_ptr = malloc(pm.bytes_to_alloc);
//     // pm.base_ptr = memory_manager->malloc(pm.bytes_to_alloc);
//     pm.base_ptr = deviceMemMalloc(memory_manager, pm.bytes_to_alloc);
//     if (!pm.base_ptr) {
//         pm.bytes_to_alloc = 0;
//         pm.base_ptr = nullptr;
//         return pm;
//     }
//     memset(pm.base_ptr, 0, pm.bytes_to_alloc);

//     // Set up pointer offsets
//     pm.n_pattern   = (int *)pm.base_ptr;
//     pm.n_items     = (Item *)(pm.n_pattern + work_item->pattern[0] + 2);
//     pm.n_start     = (int *)(pm.n_items + work_item->num_items);
//     pm.n_end       = (int *)(pm.n_start + work_item->num_transactions);
//     pm.n_utility   = (int *)(pm.n_end + work_item->num_transactions);
//     pm.tran_hash   = (Item *)(pm.n_utility + work_item->num_transactions);
//     pm.local_util  = (Item *)(pm.tran_hash + work_item->num_transactions * scale);
//     pm.subtree_util= (Item *)(pm.local_util + work_item->max_item * scale);
//     pm.work_done   = (int *)(pm.subtree_util + work_item->max_item * scale);

//     return pm;
// }

// __device__ void copyAndExtendPattern(const WorkItem *work_item, ProjectionMemory *pm) {
//     // Copy the current pattern (first n integers) and then extend it
//     memcpy(pm->n_pattern, work_item->pattern, (work_item->pattern[0] + 1) * sizeof(int));
//     pm->n_pattern[0] += 1;
//     pm->n_pattern[pm->n_pattern[0]] = work_item->primary_item;
// }



// __device__ ProjectionResult performProjection(const WorkItem *work_item, ProjectionMemory *pm) {
//     ProjectionResult res = {0, 0, 0};
//     int item_counter = 0;
//     int transaction_counter = 0;
//     int pattern_utility = 0;

//     for (int i = 0; i < work_item->num_transactions; i++) {
//         // Find the index for the primary item in this transaction.
//         int idx = binarySearchItems(work_item->items, work_item->num_items,
//                                     work_item->primary_item, work_item->start[i],
//                                     work_item->end[i] - work_item->start[i]);
//         if (idx == -1)
//             continue;

//         // Record the starting index for the new transaction
//         pm->n_start[transaction_counter] = item_counter;
//         // Combine the current transaction utility with the utility at idx.
//         pm->n_utility[transaction_counter] = work_item->utility[i] + work_item->items[idx].util;
//         pattern_utility += pm->n_utility[transaction_counter];
//         int temp_util = pm->n_utility[transaction_counter];

//         // Copy the remaining items of the transaction
//         for (int j = idx + 1; j < work_item->end[i]; j++) {
//             pm->n_items[item_counter] = work_item->items[j];
//             temp_util += work_item->items[j].util;
//             item_counter++;
//         }
//         pm->n_end[transaction_counter] = item_counter;
//         transaction_counter++;

//         // Update local utility table for all items in this transaction.
//         for (int j = idx + 1; j < work_item->end[i]; j++) {
//             uint32_t hash = hashFunction(work_item->items[j].key, work_item->max_item);
//             while (true) {
//                 if (pm->local_util[hash].key == 0 ||
//                     pm->local_util[hash].key == work_item->items[j].key)
//                 {
//                     pm->local_util[hash].key = work_item->items[j].key;
//                     pm->local_util[hash].util += temp_util;
//                     break;
//                 }
//                 hash = (hash + 1) % (work_item->max_item * scale);
//             }
//         }
//     }
//     res.item_counter = item_counter;
//     res.transaction_counter = transaction_counter;
//     res.pattern_utility = pattern_utility;
//     return res;
// }


// __device__ void storeHighUtilityPattern(const ProjectionMemory *pm, int pattern_utility,
//                              int *high_utility_patterns, int min_util) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;

//     if (pattern_utility >= min_util) {
//         // Update the count of high utility patterns.
//         // high_utility_patterns[0]++;  
//         atomicAdd(&high_utility_patterns[0], 1);
//         printf("%d|High Utility Patterns: %d\n", tid, high_utility_patterns[0]);
//         // Compute an offset to store the new pattern.
//         // int offset = (high_utility_patterns[1] += pm->n_pattern[0] + 2);
//         int offset = atomicAdd(&high_utility_patterns[1], pm->n_pattern[0] + 2);
//         for (int i = 0; i < pm->n_pattern[0]; i++) {
//             high_utility_patterns[offset + i] = pm->n_pattern[i + 1];
//         }
//         high_utility_patterns[offset + pm->n_pattern[0]] = pattern_utility;
//     }
// }

// __device__ void checkAndFreeWorkItem(CudaMemoryManager *memory_manager,WorkItem *work_item) {
//     // int ret = work_item->work_done[0]++;
//     int ret = atomicAdd(&work_item->work_done[0], 1);
//     if (ret == work_item->work_count - 1) {
//         // free(work_item->base_ptr);
//         // memory_manager->free(work_item->base_ptr);
//         deviceMemFree(memory_manager, work_item->base_ptr, work_item->bytes);
//     }
// }



// __device__ MergeResult trimMergeAndComputeSubtree(const WorkItem *work_item, ProjectionMemory *pm,
//                                          int proj_transaction_count, int min_util) {
//     MergeResult mres = {0, 0};
//     int new_item_counter = 0;
//     int new_transaction_counter = 0;
//     int mod_val = work_item->max_item * scale;

//     // Loop over all projected transactions.
//     for (int i = 0; i < proj_transaction_count; i++) {
//         int start = new_item_counter;
//         for (int j = pm->n_start[i]; j < pm->n_end[i]; j++) {
//             uint32_t hash = hashFunction(pm->n_items[j].key, mod_val);
//             while (true) {
//                 if (pm->local_util[hash].key == pm->n_items[j].key) {
//                     if (pm->local_util[hash].util >= min_util) {
//                         pm->n_items[new_item_counter] = pm->n_items[j];
//                         new_item_counter++;
//                     }
//                     break;
//                 }
//                 hash = (hash + 1) % mod_val;
//             }
//         }
//         if (start == new_item_counter)
//             continue;

//         pm->n_start[new_transaction_counter] = start;
//         pm->n_end[new_transaction_counter]   = new_item_counter;
//         pm->n_utility[new_transaction_counter] = pm->n_utility[i];


//         // pre-compute the utility of the transaction for subtree utility
//         int temp_util = pm->n_utility[new_transaction_counter];
//         for (int j = pm->n_start[new_transaction_counter]; j < pm->n_end[new_transaction_counter]; j++) {
//             temp_util += pm->n_items[j].util;
//         }

//         // Update the subtree utility table for each item in the transaction.
//         int temp = 0;
//         for (int j = pm->n_start[new_transaction_counter]; j < pm->n_end[new_transaction_counter]; j++) {
//             uint32_t hash = hashFunction(pm->n_items[j].key, mod_val);
//             while (true) {
//                 if (pm->subtree_util[hash].key == 0 ||
//                     pm->subtree_util[hash].key == pm->n_items[j].key)
//                 {
//                     pm->subtree_util[hash].key = pm->n_items[j].key;
//                     pm->subtree_util[hash].util += temp_util - temp;
//                     temp = pm->n_items[j].util;
//                     break;
//                 }
//                 hash = (hash + 1) % mod_val;
//             }
//         }


//         // Merging block: merge duplicate transactions if possible.
//         int cur_tran_idx = new_transaction_counter;
//         int tran_length = pm->n_end[cur_tran_idx] - pm->n_start[cur_tran_idx];
//         int t_hash = items_hasher(pm->n_items + start, tran_length,
//                                   work_item->num_transactions * scale);
//         bool merged = false;
//         while (true) {
//             if (pm->tran_hash[t_hash].key == 0) {
//                 pm->tran_hash[t_hash].key = t_hash;
//                 pm->tran_hash[t_hash].util = cur_tran_idx;
//                 break;
//             }
//             if (pm->tran_hash[t_hash].key == t_hash) {
//                 int existing_idx = pm->tran_hash[t_hash].util;
//                 int existing_length = pm->n_end[existing_idx] - pm->n_start[existing_idx];
//                 if (existing_length == tran_length) {
//                     bool same = true;
//                     for (int j = 0; j < tran_length; j++) {
//                         if (pm->n_items[pm->n_start[existing_idx] + j].key !=
//                             pm->n_items[pm->n_start[cur_tran_idx] + j].key)
//                         {
//                             same = false;
//                             break;
//                         }
//                     }
//                     if (same) {
//                         pm->n_utility[existing_idx] += pm->n_utility[cur_tran_idx];
//                         for (int j = 0; j < tran_length; j++) {
//                             pm->n_items[pm->n_start[existing_idx] + j].util +=
//                                 pm->n_items[pm->n_start[cur_tran_idx] + j].util;
//                         }
//                         new_item_counter -= tran_length;
//                         merged = true;
//                         break;
//                     }
//                 }
//             }
//             t_hash = (t_hash + 1) % (work_item->num_transactions * scale);
//         }
//         if (!merged)
//             new_transaction_counter++;
//     }

//     mres.new_item_counter = new_item_counter;
//     mres.new_transaction_counter = new_transaction_counter;
//     return mres;
// }

// __device__ int countSubtreeUtility(const WorkItem *work_item, const ProjectionMemory *pm, int min_util) {
//     int counter = 0;
//     for (int i = 0; i < work_item->max_item * scale; i++) {
//         if (pm->subtree_util[i].util >= min_util)
//             counter++;
//     }
//     return counter;
// }

// __device__ void pushNewWorkItems(AtomicWorkStack *q, const WorkItem *old_work_item,
//                       ProjectionMemory *pm, int new_item_counter, int new_transaction_counter,
//                       int subtree_util_counter, int min_util) {
//     // Compute the number of items that meet the local utility threshold.
//     int local_util_counter = 0;
//     for (int i = 0; i < old_work_item->max_item * scale; i++) {
//         if (pm->local_util[i].util >= min_util)
//             local_util_counter++;
//     }


//     for (int i = 0; i < old_work_item->max_item * scale; i++) {
//         if (pm->subtree_util[i].util < min_util)
//             continue;

//         WorkItem new_work_item;
//         new_work_item.pattern         = pm->n_pattern;
//         new_work_item.items           = pm->n_items;
//         new_work_item.num_items       = new_item_counter;
//         new_work_item.start           = pm->n_start;
//         new_work_item.end             = pm->n_end;
//         new_work_item.utility         = pm->n_utility;
//         new_work_item.num_transactions= new_transaction_counter;
//         new_work_item.primary_item    = pm->subtree_util[i].key;
//         new_work_item.max_item        = local_util_counter;
//         new_work_item.work_done       = pm->work_done;
//         new_work_item.work_count      = subtree_util_counter;
//         new_work_item.bytes           = pm->bytes_to_alloc;
//         new_work_item.base_ptr        = pm->base_ptr;

//         // stack_push(q, new_work_item);
//         while (!stack_push(q, new_work_item)) {}
//     }
//     atomicAdd(&q->active, subtree_util_counter);
// }


// __global__ void mine(CudaMemoryManager *memory_manager, AtomicWorkStack *work_queue, int min_util, int *high_utility_patterns)
// {
//     WorkItem work_item;
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;

//     while (stack_get_work_count(work_queue) > 0)
//     {
//         if (!stack_pop(work_queue, &work_item)) 
//         {
//             // __nanosleep(100);
//             __threadfence_system();
//             continue; // queue is momentarily empty
//         }
//         printf("%d:Queue Size: %d\n", tid, stack_get_work_count(work_queue));


//         // 1. Allocate memory for projection.
//         ProjectionMemory pm = allocateProjectionMemory(memory_manager, &work_item);
//         if (pm.base_ptr == nullptr) {
//             atomicSub(&work_queue->active, 1);
//             // work_queue->active--;
//             // __threadfence_system();

//             // push the work item back to the queue
//             while (!stack_push(work_queue, work_item)) {}
//             atomicAdd(&work_queue->active, 1);


//             printf("Failed to allocate memory for projection\n");
//             continue;
//         }

//         // int *slowdown = (int *)deviceMemMalloc(memory_manager, sizeof(int));
//         // memset(slowdown, 0, sizeof(int));

//         // printf("%d:Slowdown: %d\n", tid, *slowdown);

//         // slowdowner<<<4, 1>>>(slowdown);

//         // while (*slowdown < 4)
//         // {
//         //     // __nanosleep(100);
//         //     __threadfence_system();
//         // }
//         // printf("%d:Slowdown: %d\n", tid, *slowdown);

//         // deviceMemFree(memory_manager, slowdown, sizeof(int));



//         // 2. Copy and extend the pattern.
//         copyAndExtendPattern(&work_item, &pm);

//         // 3. Perform projection.
//         // printf("%d:Performing Projection\n", tid);
//         ProjectionResult proj = performProjection(&work_item, &pm);

//         // 4. Store high utility pattern if threshold met.
//         storeHighUtilityPattern(&pm, proj.pattern_utility, high_utility_patterns, min_util);


//         // If no valid transactions were found, free and update work_done.
//         if (proj.transaction_counter == 0) {
//             atomicSub(&work_queue->active, 1);
//             // printf("No valid transactions found\n");
//             // memory_manager->free(pm.base_ptr);
//             deviceMemFree(memory_manager, pm.base_ptr, pm.bytes_to_alloc);
//             checkAndFreeWorkItem(memory_manager, &work_item);
//             continue;
//         }


//         // printf("%d:Trimming and Merging\n", tid);
//         MergeResult mergeRes = trimMergeAndComputeSubtree(&work_item, &pm,
//                                                           proj.transaction_counter, min_util);

//         int subtree_util_counter = countSubtreeUtility(&work_item, &pm, min_util);


//         // 7. Create and push new work items if there is any valid subtree.
//         // printf("%d:Pushing New Work Items\n", tid);
//         if (subtree_util_counter) {
//             pushNewWorkItems(work_queue, &work_item, &pm, mergeRes.new_item_counter,
//                              mergeRes.new_transaction_counter, subtree_util_counter, min_util);
//         } else {
//             // memory_manager->free(pm.base_ptr);
//             deviceMemFree(memory_manager, pm.base_ptr, pm.bytes_to_alloc);
//         }

//         // printf("%d:Work Done\n", tid);
//         // // 8. Update work_done and free memory if all work is done.
//         checkAndFreeWorkItem(memory_manager, &work_item);

//         // printf("\n");
//         atomicSub(&work_queue->active, 1);
//         // __threadfence_system();
//     }
//     printf("Thread %d finished\n", tid);
// }
