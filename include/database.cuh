#pragma once
#include <cstdint>


struct Item {
    int key;
    int util;
} __attribute__((packed)); // Ensures tight memory layout

struct Transaction {
    const Item* data;
    int utility;
    int start;
    int end;
    int hash;

    __device__ __host__
    int length() const { return end - start; }

    __device__ __host__
    const Item* get() const { return data + start; }

    __device__ int findItem(uint32_t search_id) const {
        int l = start, r = end - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (data[mid].key == search_id)
                return mid;
            data[mid].key < search_id ? l = mid + 1 : r = mid - 1;
        }
        return -1;
    }
};

struct Database {
    Item* d_data;
    Transaction* d_transactions;
    int numTransactions;
    int transaction_tracker;
    int numItems;
};