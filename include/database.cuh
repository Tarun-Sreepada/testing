#pragma once
#include <cstdint>

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

struct Item
{
    int key;
    int util;
};

__device__ uint32_t items_hasher(const Item *items, int n, int tableSize)
{
    uint32_t hash = 0;
    for (int i = 0; i < n; i++)
        hash ^= pcg_hash(items[i].key + i);
    return hash % tableSize;
}

/*
Item *items: array of items
int n: size of the array
int key: key to search
*/
__device__ int find_item(const Item *items, int n, int key)
{
    uint32_t hash = hashFunction(key, n);

    while (true)
    {
        if (items[hash].key == key)
            return hash;
        if (items[hash].key == 0)
            return -1;
        hash = (hash + 1) % n;
    }
}

struct Transaction
{
    // Pointer to the first Item of this transaction’s subarray.
    Item* data;
    // Number of items in this transaction.
    int length;
    int utility;

    __device__ __host__ int size() const { return length; }

    // Allow treating the transaction like an array.
    __device__ __host__ Item &operator[](int i) { return data[i]; }
    
    __device__ __host__ Item *operator->() { return data; }
    
    __device__ __host__ Item *get() { return data; }

    // Binary search for a given key in this transaction.
    __device__ int findItem(uint32_t search_id) const
    {
        int l = 0, r = length - 1;
        while (l <= r)
        {
            int mid = l + (r - l) / 2;
            if (data[mid].key == search_id)
                return mid;
            data[mid].key < search_id ? l = mid + 1 : r = mid - 1;
        }
        return -1;
    }
};



struct Database
{
    Item *d_data;
    Transaction *d_transactions;
    int numTransactions;
    int transaction_tracker;
    int numItems;

    __device__ bool sameKey(int t1, int t2)
    {
        const Transaction &trans1 = d_transactions[t1];
        const Transaction &trans2 = d_transactions[t2];

        if (trans1.length != trans2.length)
            return false;

        for (int i = 0; i < trans1.length; i++)
        {
            if (trans1.data[i].key != trans2.data[i].key)
                return false;
        }
        return true;
    }
};

__device__ __host__ void printDatabase(const Database *db)
{
    if (!db) {
        printf("Database pointer is NULL!\n");
        return;
    }
    
    for (int t = 0; t < db->numTransactions; t++)
    {
        const Transaction &tran = db->d_transactions[t];
        
        printf("%d|", tran.utility);
        for (int i = 0; i < tran.length; i++)
        {
            printf("%d:%d ", tran.data[i].key, tran.data[i].util);
        }
        printf("\n");
    }
}