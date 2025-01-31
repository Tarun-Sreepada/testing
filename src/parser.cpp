#include "parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Implementation of the split function
std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t pos_start = 0, pos_end;
    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        if (pos_end > pos_start) {
            tokens.emplace_back(s.substr(pos_start, pos_end - pos_start));
        }
        pos_start = pos_end + delimiter.length();
    }
    if (pos_start < s.length()) {
        tokens.emplace_back(s.substr(pos_start));
    }
    return tokens;
}

// Implementation of the read_file function
ReadFileResult read_file(const std::string& inputFile, const std::string& sep, int minUtil) {
    ReadFileResult result;
    // Temporary data structures
    std::vector<std::pair<std::vector<std::string>, std::vector<int>>> fileData;
    // TWU dictionary: item string -> total weight
    std::unordered_map<std::string, int> twu;

    std::ifstream infile(inputFile);
    if (!infile) {
        std::cerr << "Error opening file: " << inputFile << "\n";
        exit(1);
    }
    std::string line;
    while (std::getline(infile, line)) {
        // Expected format: items : weight : utility_list
        // Items and utility_list are further separated by sep.
        std::vector<std::string> parts = split(line, ":");
        // print parts
        if (parts.size() < 3) {
            std::cerr << "Invalid line: " << line << "\n";
            continue;
        }


        std::vector<std::string> items = split(parts[0], sep);

        int weight;
        try {
            weight = std::stoi(parts[1]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid weight in line: " << line << "\n";
            continue;
        }

        std::vector<std::string> utilStrs = split(parts[2], sep);

        std::vector<int> utils;
        bool valid = true;
        for (const auto& s : utilStrs) {
            try {
                utils.push_back(std::stoi(s));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid utility value '" << s << "' in line: " << line << "\n";
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        if (items.size() != utils.size()) {
            std::cerr << "Mismatch between number of items and utilities in line: " << line << "\n";
            continue;
        }

        fileData.emplace_back(std::make_pair(items, utils));

        // Update TWU: add weight for each item
        for (const auto& item : items) {
            twu[item] += weight;
        }
    }
    infile.close();

    // Filter TWU based on minUtil threshold.
    for (auto it = twu.begin(); it != twu.end();) {
        if (it->second < minUtil) {
            it = twu.erase(it);
        } else {
            ++it;
        }
    }

    // Create a sorted vector of (item, utility) in descending order by utility.
    std::vector<std::pair<std::string, int>> sortedTWU(twu.begin(), twu.end());
    std::sort(sortedTWU.begin(), sortedTWU.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Map each item (string) to an integer (starting from count downwards).
    std::unordered_map<std::string, int> strToInt;
    int t = static_cast<int>(sortedTWU.size());
    result.max_item = t;
    for (const auto& p : sortedTWU) {
        strToInt[p.first] = t;
        result.rename[t] = p.first;
        t--;
    }

    // Build filtered transactions and compute subtree utility.
    std::unordered_map<int, int> subtree;
    std::map<std::vector<int>, std::vector<int>> filteredTransactions;
    for (const auto& entry : fileData) {
        const std::vector<std::string>& items = entry.first;
        const std::vector<int>& utils = entry.second;
        std::vector<std::pair<int, int>> transaction;
        for (size_t i = 0; i < items.size(); ++i) {
            auto it = strToInt.find(items[i]);
            if (it != strToInt.end()) {
                transaction.emplace_back(std::make_pair(it->second, utils[i]));
            }
        }
        if (transaction.empty()) {
            continue;
        }
        // Sort transaction by item id.
        std::sort(transaction.begin(), transaction.end(),
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first < b.first; });

        std::vector<int> key;
        std::vector<int> val;
        for (const auto& p : transaction) {
            key.emplace_back(p.first);
            val.emplace_back(p.second);
        }

        // Use key (sorted vector) as key for filteredTransactions.
        auto it = filteredTransactions.find(key);
        if (it == filteredTransactions.end()) {
            filteredTransactions[key] = val;
        } else {
            // If the transaction already exists, add corresponding utilities.
            for (size_t i = 0; i < val.size(); ++i) {
                it->second[i] += val[i];
            }
        }

        // Compute subtree utility for this transaction.
        int subUtil = 0;
        for (int v : val)
            subUtil += v;
        int temp = 0;
        for (size_t i = 0; i < key.size(); ++i) {
            subtree[key[i]] += subUtil - temp;
            temp += val[i];
        }
    }

    // Determine primary items: those with subtree utility >= minUtil.
    for (const auto& p : subtree) {
        if (p.second >= minUtil) {
            result.primary.emplace_back(static_cast<uint32_t>(p.first));
        }
    }

    result.filteredTransactions = std::move(filteredTransactions);

    return result;
}
