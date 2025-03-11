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
    int value;
};

struct Utils
{
    int key;
    int local_util;
    int subtree_util;
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
__device__ int find_item(const Utils *lu_su, int n, int key)
{
    uint32_t hash = hashFunction(key, n);

    while (true)
    {
        if (lu_su[hash].key == key)
            return hash;
        if (lu_su[hash].key == 0)
            return -1;
        hash = (hash + 1) % n;
    }
}

struct Transaction
{
    // Pointer to the first Item of this transaction’s subarray.
    Item *data;
    // Number of items in this transaction.
    int length;
    int utility;

    __device__ __host__ int size() const { return length; }

    // Allow treating the transaction like an array.
    __device__ __host__ Item &operator[](int i) { return data[i]; }

    __device__ __host__ Item *operator->() { return data; }

    __device__ __host__ Item *get() { return data; }

    __device__ __forceinline__ int findItem(uint32_t search_id) const
    {
        if (length == 0)
            return -1; // Early exit if no elements

        int l = 0, r = length - 1;
        uint32_t firstKey = data[l].key;
        uint32_t lastKey = data[r].key;

        // Early boundary checks
        if (firstKey > search_id || lastKey < search_id)
            return -1;
        if (firstKey == search_id)
            return l;
        if (lastKey == search_id)
            return r;

        while (l <= r)
        {
            int mid = (l + r) >> 1;          // Bit-shift division
            uint32_t midKey = data[mid].key; // Load once into register
            if (midKey == search_id)
                return mid;
            else if (midKey < search_id)
                l = mid + 1;
            else
                r = mid - 1;
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

    // = operator
    __device__ Database &operator=(const Database &other)
    {
        d_data = other.d_data;
        d_transactions = other.d_transactions;
        numTransactions = other.numTransactions;
        transaction_tracker = other.transaction_tracker;
        numItems = other.numItems;
        return *this;
    }

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
    if (!db)
    {
        printf("Database pointer is NULL!\n");
        return;
    }

    for (int t = 0; t < db->numTransactions; t++)
    {
        const Transaction &tran = db->d_transactions[t];

        printf("%d|", tran.utility);
        for (int i = 0; i < tran.length; i++)
        {
            printf("%d:%d ", tran.data[i].key, tran.data[i].value);
        }
        printf("\n");
    }
}