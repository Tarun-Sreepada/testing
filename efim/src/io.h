#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <stdint.h>

// This structure holds an item id and its utility.
struct Item {
    uint32_t item;
    uint32_t util;
};

// Splits a string (by a given delimiter).
std::vector<std::string> split(const std::string &str, char delimiter);

// Reads the transaction database from file and returns the following items as a tuple:
//   - vector<Item> : all items (concatenated from all transactions)
//   - vector<uint32_t> : start indices of each transaction in the items vector
//   - vector<uint32_t> : end indices of each transaction in the items vector
//   - vector<uint32_t> : subtree utilities per item id
//   - map<uint32_t, uint32_t> : filtered transaction-weight utilities (TWU)
//   - uint32_t : maximum item id in the dataset
std::tuple<std::vector<Item>, std::vector<uint32_t>, std::vector<uint32_t>,
           std::vector<uint32_t>, std::map<uint32_t, uint32_t>, uint32_t>
readFile(const std::string &filename, char delimiter, uint32_t minUtil);

#endif // IO_UTILS_H
