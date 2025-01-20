#include "io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <stdlib.h>


/// Helper: tokenize a string_view by a delimiter without allocations.
static inline void fastTokenize(const std::string_view sv, char delim, std::vector<std::string_view> &tokens) {
    tokens.clear();
    size_t start = 0;
    while (start < sv.size()) {
        size_t pos = sv.find(delim, start);
        if (pos == std::string_view::npos)
            pos = sv.size();
        if (pos > start)
            tokens.push_back(sv.substr(start, pos - start));
        start = pos + 1;
    }
}

/// Reads entire file into a single string.
static inline std::string readWholeFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << "\n";
        exit(1);
    }
    size_t fileSize = file.tellg();
    std::string content;
    content.resize(fileSize);
    file.seekg(0);
    file.read(&content[0], fileSize);
    return content;
}

/// Optimized version of readFile that minimizes temporary allocations.
std::tuple<std::vector<Item>, std::vector<uint32_t>, std::vector<uint32_t>,
           std::vector<uint32_t>, std::map<uint32_t, uint32_t>, uint32_t>
readFile(const std::string &filename, char delimiter, uint32_t minUtil)
{
    // Reserve some capacity for transactions if possible.
    std::vector<std::vector<Item>> transactions;
    transactions.reserve(10000);

    std::map<uint32_t, uint32_t> twu; // transaction-weight utility for each item

    // Read entire file into memory.
    std::string fileContent = readWholeFile(filename);
    std::vector<std::string_view> fileLines;
    {
        // Split file content into lines without reallocating string buffers.
        size_t pos = 0;
        while (pos < fileContent.size()) {
            size_t eol = fileContent.find('\n', pos);
            if (eol == std::string::npos)
                eol = fileContent.size();
            if (eol > pos) {  // non-empty line
                fileLines.push_back(std::string_view(fileContent.c_str() + pos, eol - pos));
            }
            pos = eol + 1;
        }
    }

    // Temporary vector to hold tokens.
    std::vector<std::string_view> tokens;
    tokens.reserve(10);

    for (const std::string_view &line : fileLines) {
        // Expected format: items:transactionWeight:utilities
        tokens.clear();
        fastTokenize(line, ':', tokens);
        if (tokens.size() < 3) {
            std::cerr << "Invalid line: " << line << "\n";
            continue;
        }
        // tokens[0] holds items, tokens[1] holds transaction weight, tokens[2] holds utilities

        // Parse transaction weight
        uint32_t transactionWeight = 0;
        try {
            transactionWeight = static_cast<uint32_t>(std::stoi(std::string(tokens[1])));
        }
        catch (const std::exception &e) {
            std::cerr << "Invalid transaction weight in line: " << line << "\nException: " << e.what() << "\n";
            continue;
        }

        // Split tokens[0] and tokens[2] using the given delimiter.
        std::vector<std::string_view> itemTokens;
        std::vector<std::string_view> utilTokens;
        fastTokenize(tokens[0], delimiter, itemTokens);
        fastTokenize(tokens[2], delimiter, utilTokens);
        if (itemTokens.size() != utilTokens.size()) {
            std::cerr << "Mismatch between items and utilities in line: " << line << "\n";
            continue;
        }
        std::vector<Item> transaction;
        transaction.reserve(itemTokens.size());
        for (size_t i = 0; i < itemTokens.size(); ++i) {
            uint32_t itemId = 0;
            uint32_t util = 0;
            try {
                // Use std::string_view->string conversion only when parsing integers.
                itemId = static_cast<uint32_t>(std::stoi(std::string(itemTokens[i])));
                util = static_cast<uint32_t>(std::stoi(std::string(utilTokens[i])));
            }
            catch (const std::exception &e) {
                std::cerr << "Error parsing item or utility in line: " << line << "\nException: " << e.what() << "\n";
                continue;
            }
            transaction.push_back({itemId, util});
            // Update TWU by adding the transaction weight.
            twu[itemId] += transactionWeight;
        }
        transactions.push_back(std::move(transaction));
    }

    // Filter TWU based on minimum utility.
    uint32_t maxItem = 0;
    std::map<uint32_t, uint32_t> filtered_twu;
    for (const auto &p : twu) {
        maxItem = std::max(maxItem, p.first);
        if (p.second >= minUtil)
            filtered_twu[p.first] = p.second;
    }

    // (Optional) sort filtered TWU items by utility descending.
    std::vector<std::pair<uint32_t, uint32_t>> twuVec(filtered_twu.begin(), filtered_twu.end());
    std::sort(twuVec.begin(), twuVec.end(),
              [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
                  return a.second > b.second;
              });

    // Compute subtree utilities from transactions.
    std::vector<uint32_t> subtreeUtility(maxItem + 1, 0);
    for (auto &transaction : transactions) {
        // Erase items that are not frequent according to filtered TWU.
        transaction.erase(std::remove_if(transaction.begin(), transaction.end(),
                                         [&filtered_twu](const Item &item) {
                                             return filtered_twu.find(item.item) == filtered_twu.end();
                                         }),
                          transaction.end());
        // Sort the transaction by TWU value (ascending order so that smaller utility items come first).
        std::sort(transaction.begin(), transaction.end(),
                  [&filtered_twu](const Item &a, const Item &b) {
                      return filtered_twu.at(a.item) < filtered_twu.at(b.item);
                  });

        uint32_t sum = std::accumulate(transaction.begin(), transaction.end(), 0,
                                       [](uint32_t acc, const Item &item) { return acc + item.util; });
        for (const auto &item : transaction) {
            subtreeUtility[item.item] += sum;
            sum -= item.util;
        }
    }

    // Build contiguous arrays to represent transactions.
    std::vector<uint32_t> start;
    std::vector<uint32_t> end;
    std::vector<Item> items;

    // Reserve estimated capacity if possible.
    start.reserve(transactions.size());
    end.reserve(transactions.size());
    items.reserve(std::accumulate(transactions.begin(), transactions.end(), size_t(0),
                                  [](size_t s, const std::vector<Item> &tr) { return s + tr.size(); }));

    uint32_t curr = 0;
    for (const auto &transaction : transactions) {
        start.push_back(curr);
        for (const auto &it : transaction) {
            items.push_back(it);
            ++curr;
        }
        end.push_back(curr);
    }

    return {std::move(items), std::move(start), std::move(end),
            std::move(subtreeUtility), std::move(filtered_twu), maxItem};
}