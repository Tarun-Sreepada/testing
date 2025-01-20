#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <stdint.h>
#include <cuda_runtime.h>
#include "io.h"
#include "mine.cuh"
#include "allocator.cuh"

#define KILO 1024ULL
#define MEGA KILO * KILO
#define GIGA KILO * MEGA



int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <delimiter> <minUtil>" << std::endl;
        return 1;
    }

    std::string filename(argv[1]);
    char delimiter;
    uint32_t minUtil;

    try {
        if (strcmp(argv[2], "\\s") == 0)
            delimiter = ' ';
        else if (strcmp(argv[2], "\\t") == 0)
            delimiter = '\t';
        else
            delimiter = argv[2][0];

        minUtil = std::stoi(argv[3]);
    } catch (const std::exception &e) {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    // Read the file and create the transaction database.
    auto [items, start, end, subtreeUtil, filtered_twu, maxItem] = readFile(filename, delimiter, minUtil);

    std::cout << "File read successfully." << std::endl;


    // Build the primary and secondary arrays.
    std::vector<uint32_t> primary(maxItem + 1, 0);
    for (size_t i = 0; i < primary.size(); ++i)
    {
        if (subtreeUtil[i] >= minUtil)
            primary[i] = i;
        else
            primary[i] = 0;
    }

    std::vector<uint32_t> secondary(maxItem + 1, 0);
    for (size_t i = 0; i < secondary.size(); ++i)
    {
        // Here we use filtered_twu as a placeholder for local utility.
        if (filtered_twu.find(i) != filtered_twu.end() && filtered_twu.at(i) >= minUtil)
            secondary[i] = i;
        else
            secondary[i] = 0;
    }

    // // Allocate device memory.
    BumpAllocator *alloc = createUnifiedBumpAllocator(25 * GIGA);
    if (!alloc)
    {
        std::cerr << "Error creating bump allocator." << std::endl;
        return 1;
    }

    // print alloc bytes used
    double mb_used = static_cast<double>(alloc->offset) / static_cast<double>(MEGA);
    std::cout << "Bytes used by the allocator: " << alloc->offset << "\t(MB: " << mb_used << ")" << std::endl;

    uint32_t pattern_init = 0;
    uint32_t *d_pattern = bump_allocate_and_copy(alloc, &pattern_init, 1);
    if (!d_pattern)
    {
        std::cerr << "Error allocating pattern memory." << std::endl;
        return 1;
    }

     mb_used = static_cast<double>(alloc->offset) / static_cast<double>(MEGA);
    std::cout << "Bytes used by the allocator: " << alloc->offset << "\t(MB: " << mb_used << ")" << std::endl;
    

    // Allocate the start, end, primary, secondary, items, and utility arrays.

    uint32_t *d_start = bump_allocate_and_copy(alloc, start.data(), start.size());
    uint32_t *d_end = bump_allocate_and_copy(alloc, end.data(), end.size());
    uint32_t *d_primary = bump_allocate_and_copy(alloc, primary.data(), primary.size());
    uint32_t *d_secondary = bump_allocate_and_copy(alloc, secondary.data(), secondary.size());
    Item *d_items = bump_allocate_and_copy(alloc, items.data(), items.size());
    uint32_t *d_utility = bump_allocate_and_copy(alloc, subtreeUtil.data(), start.size());

     mb_used = static_cast<double>(alloc->offset) / static_cast<double>(MEGA);
    std::cout << "Bytes used by the allocator: " << alloc->offset << "\t(MB: " << mb_used << ")" << std::endl;



    uint32_t *d_pattern_counter;
    cudaMallocManaged(&d_pattern_counter, sizeof(uint32_t));
    std::cout << "Pattern counter: " << d_pattern_counter[0] << std::endl;


    if (!d_start || !d_end || !d_primary || !d_secondary || !d_items || !d_utility)
    {
        std::cerr << "Error allocating device memory." << std::endl;
        return 1;
    }

    uint32_t numTransactions = start.size();

    // Copy the data to device memory.
    cudaMemcpy(d_start, start.data(), start.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end.data(), end.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primary, primary.data(), primary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_secondary, secondary.data(), secondary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_items, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice);
    cudaMemset(d_utility, 0, start.size() * sizeof(uint32_t));

  
    mine<<<maxItem + 1, 1>>>(alloc, d_pattern, d_items, items.size(), d_start, d_end,
                             d_utility, start.size(), d_primary, d_secondary,
                             maxItem, minUtil, d_pattern_counter);
    cudaDeviceSynchronize();

    std::cout << "Pattern counter: " << d_pattern_counter[0] << std::endl;

    // print bytes used by the allocator
    mb_used = static_cast<double>(alloc->offset) / static_cast<double>(MEGA);
    std::cout << "Bytes used by the allocator: " << alloc->offset << "\t(MB: " << mb_used << ")" << std::endl;

    // free device memory
    freeUnifiedBumpAllocator(alloc);

    return 0;
}
