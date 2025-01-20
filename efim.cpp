#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <unordered_map>

// Helper: split a string by a delimiter.
std::vector<std::string> split(const std::string &str, const std::string &delimiter) {
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos) {
        tokens.push_back(str.substr(prev, pos - prev));
        prev = pos + delimiter.length();
    }
    tokens.push_back(str.substr(prev));
    return tokens;
}

// Binary search function returning the index of x in arr,
// or -1 if not found.
int binarySearch(const std::vector<int> &arr, int x) {
    int l = 0;
    int r = static_cast<int>(arr.size()) - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        else if (arr[mid] < x)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return -1;
}

// A structure to hold transactions.
struct Transaction {
    std::vector<int> keys; // sorted items (represented by int)
    std::vector<int> vals; // corresponding utility values
    int acc;               // accumulated utility
};

// For ordering keys in maps (we use vector<int> as a key).
struct VectorComparator {
    bool operator()(const std::vector<int> &a, const std::vector<int> &b) const {
        return a < b; // lexicographical compare provided by std::vector
    }
};

class efim {
public:
    efim(const std::string &iFile, int minUtil, const std::string &sep = "\t", int threads = 1)
        : inputFile(iFile),
          minUtil(minUtil),
          sep(sep),
          threads(threads),
          runtime(0) { }

    // Starts the mining process
    void startMine() {
        mine();
    }

    // Perform the mining algorithm.
    void mine() {
        auto startTime = std::chrono::steady_clock::now();

        // Read and process file to get filtered transactions and primary items.
        std::map<std::vector<int>, Transaction, VectorComparator> transactions;
        std::vector<int> primary;
        _read_file(transactions, primary);

        // Begin recursive search
        std::vector<int> prefix;
        _search(prefix, transactions, primary);

        auto endTime = std::chrono::steady_clock::now();
        runtime = std::chrono::duration<double>(endTime - startTime).count();
    }

    // Save results to a file.
    void save(const std::string &outFile) {
        std::ofstream writer(outFile);
        if (!writer) {
            std::cerr << "Error opening output file: " << outFile << "\n";
            return;
        }
        // Here we write the discovered patterns (stored in Patterns)
        for (const auto &entry : Patterns) {
            writer << entry.first << ":" << entry.second << "\n";
        }
        writer.close();
    }

    // Print final results.
    void printResults() const {
        std::cout << "Total number of High Utility Patterns: " << Patterns.size() << "\n";
        std::cout << "Total Execution Time (seconds): " << runtime << "\n";
        // Memory info is platform‐dependent in C++ so is omitted here.
    }

    // Get discovered patterns.
    const std::map<std::string, int>& getPatterns() const {
        return Patterns;
    }

    double getRuntime() const { return runtime; }

private:
    std::string inputFile;
    int minUtil;
    std::string sep;
    int threads;

    // Mapping from pattern string to total utility support.
    std::map<std::string, int> Patterns;
    // Mapping from int (item id) back to original string.
    std::unordered_map<int, std::string> rename;

    double runtime;

    // _read_file: Reads the input file, calculates Transaction Weighted Utility (TWU),
    // assigns integer IDs to items, filters transactions, and returns:
    //    transactions: a map from sorted vector of item IDs to a Transaction
    //    primary: a vector of item IDs with subtree utility >= minUtil
    void _read_file(std::map<std::vector<int>, Transaction, VectorComparator> &filteredTransactions,
                    std::vector<int> &primary) {
        // File data: each entry is a pair: (vector of item strings, vector of utility ints)
        std::vector<std::pair<std::vector<std::string>, std::vector<int>>> fileData;
        // TWU dictionary: item string -> total weight
        std::unordered_map<std::string, int> twu;

        std::ifstream infile(inputFile);
        if (!infile) {
            std::cerr << "Error opening file: " << inputFile << "\n";
            return;
        }
        std::string line;
        while (std::getline(infile, line)) {
            // Expected format: items : weight : utility_list
            // Items and utility_list are further separated by sep.
            std::vector<std::string> parts = split(line, ":");
            if (parts.size() < 3)
                continue;

            // std::cout << std::endl;
            std::vector<std::string> items = split(parts[0], sep);


            // // print all items
            // for (const auto &i : items) {
            //     std::cout << i << " ";
            // }
            // std::cout << std::endl;


            int weight = std::stoi(parts[1]);
            std::vector<std::string> utilStrs = split(parts[2], sep);


            // // print all utilStrs
            // for (const auto &u : utilStrs) {
            //     std::cout << u << " ";
            // }
            // std::cout << std::endl;

            // std::cout << "weight: " << weight << std::endl;


            std::vector<int> utils;
            for (const auto &s : utilStrs)
                utils.push_back(std::stoi(s));
            fileData.push_back({items, utils});
            // Update TWU: add weight for each item
            for (const auto &item : items) {
                twu[item] += weight;
            }
        }
        infile.close();

        // // print all twu
        // std::cout << "TWU: " << std::endl;
        // for (const auto &t : twu) {
        //     std::cout << t.first << ":" << t.second << std::endl;
        // }


        // Filter twu based on minUtil threshold.
        for (auto it = twu.begin(); it != twu.end(); ) {
            if (it->second < minUtil)
                it = twu.erase(it);
            else
                ++it;
        }

     
        // Create a sorted vector of (item, utility) in descending order by utility.
        std::vector<std::pair<std::string, int>> sortedTWU(twu.begin(), twu.end());
        std::sort(sortedTWU.begin(), sortedTWU.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        // Map each item (string) to an integer (starting from count downwards).
        std::unordered_map<std::string, int> strToInt;
        int t = static_cast<int>(sortedTWU.size());
        for (const auto &p : sortedTWU) {
            strToInt[p.first] = t;
            rename[t] = p.first;
            t--;
        }

        // Build filtered transactions and compute subtree utility.
        std::unordered_map<int, int> subtree;
        for (const auto &entry : fileData) {
            const std::vector<std::string> &items = entry.first;
            const std::vector<int> &utils = entry.second;
            std::vector<std::pair<int, int>> transaction;
            for (size_t i = 0; i < items.size(); ++i) {
                if (strToInt.find(items[i]) != strToInt.end()) {
                    transaction.push_back({strToInt[items[i]], utils[i]});
                }
            }
            if (transaction.empty())
                continue;
            // Sort transaction by item id.
            std::sort(transaction.begin(), transaction.end(),
                      [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.first < b.first; });

            std::vector<int> key;
            std::vector<int> val;
            for (const auto &p : transaction) {
                key.push_back(p.first);
                val.push_back(p.second);
            }

            // Use key (sorted vector) as key for filteredTransactions.
            if (filteredTransactions.find(key) == filteredTransactions.end()) {
                filteredTransactions[key] = Transaction{key, val, 0};
            } else {
                // If the transaction already exists, add corresponding utilities.
                Transaction &trans = filteredTransactions[key];
                for (size_t i = 0; i < val.size(); ++i)
                    trans.vals[i] += val[i];
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

        // // print all transactions
        // std::cout << "Transactions: " << std::endl;
        // for (const auto &t : filteredTransactions) {
        //     std::cout << "key: ";
        //     for (const auto &k : t.first) {
        //         std::cout << k << " ";
        //     }
        //     std::cout << "val: ";
        //     for (const auto &v : t.second.vals) {
        //         std::cout << v << " ";
        //     }
        //     std::cout << std::endl;
        //     std::cout << "acc: " << t.second.acc << std::endl;
        // }


        // Determine primary items: those with subtree utility >= minUtil.
        for (const auto &p : subtree) {
            if (p.second >= minUtil)
                primary.push_back(p.first);
        }

        // print all primary
        // std::cout << "Primary: " << std::endl;
        // for (const auto &p : primary) {
        //     std::cout << p << " ";
        // }
        // std::cout << std::endl;
    }

    // _search: Recursive search for high utility patterns.
    //
    // Parameters:
    //   prefix: the current prefix (vector of item ids)
    //   transactions: current set of transactions (keyed by sorted vector of item ids)
    //   primary: list of candidate items (item ids) to extend the prefix
   void _search(const std::vector<int> &prefix,
             const std::map<std::vector<int>, Transaction, VectorComparator> &transactions,
             const std::vector<int> &primary) {
    for (int item : primary) {
        // newTransactions will store the next-level transactions.
        std::map<std::vector<int>, Transaction, VectorComparator> nTransactions;
        // local utility counter for items appearing in new transactions.
        std::unordered_map<int, int> localUtil;

        // print all prefix for debugging
        // std::cout << "Prefix: ";
        // for (const auto &p : prefix) {
        //     std::cout << p << " ";
        // }
        // std::cout << item << std::endl;

        // Iterate through each current transaction.
        for (const auto &entry : transactions) {
            const Transaction &trans = entry.second;
            // Process transactions that contain the current candidate item.
            // (Since trans.keys is sorted, we use binary search.)
            if (std::find(trans.keys.begin(), trans.keys.end(), item) != trans.keys.end()) {
                int pos = binarySearch(trans.keys, item) + 1; // position after the found item

                // Build a new transaction from the remaining items.
                std::vector<int> newKeys(trans.keys.begin() + pos, trans.keys.end());
                std::vector<int> newVals(trans.vals.begin() + pos, trans.vals.end());
                // Note: Even if newKeys is empty, we still want to record the utility.
                Transaction newTrans;
                newTrans.keys = newKeys;
                newTrans.vals = newVals;
                // Add the utility from the item at position pos-1.
                newTrans.acc = trans.acc + trans.vals[pos - 1];

                // Merge newTrans into nTransactions.
                auto it = nTransactions.find(newKeys);
                if (it == nTransactions.end()) {
                    nTransactions[newKeys] = newTrans;
                } else {
                    Transaction &existing = it->second;
                    for (size_t i = 0; i < newVals.size(); ++i) {
                        existing.vals[i] += newVals[i];
                    }
                    existing.acc += newTrans.acc;
                }

                // Calculate total utility of this new transaction.
                int tranTotal = newTrans.acc;
                for (int v : newVals)
                    tranTotal += v;
                // Update local utility for each item in newKeys.
                for (int key : newKeys)
                    localUtil[key] += tranTotal;
            }
        } // end for each transaction

        // Sum total utility from nTransactions.
        int total = 0;
        for (const auto &p : nTransactions) {
            total += p.second.acc;
        }

        // std::cout << "Total Util: " << total << std::endl;

        // // Print all nTransactions for debugging.
        // std::cout << "New Transactions: " << std::endl;
        // for (const auto &n : nTransactions) {
        //     for (int i = 0; i < n.first.size(); i++) {
        //         std::cout << rename[n.first[i]] << ":" << n.second.vals[i] << " ";
        //     }
        //     std::cout << "||" << n.second.acc << std::endl;
        // }

        // If total utility meets the threshold, record the pattern.
        if (total >= minUtil) {
            // Build pattern string from prefix + current item.
            std::string pattern;
            for (int id : prefix)
                pattern += rename[id] + "\t";
            pattern += rename[item];
            Patterns[pattern] = total;
        }

        // // Print local utilities for debugging.
        // std::cout << "Local Util: " << std::endl;
        // for (const auto &l : localUtil) {
        //     std::cout << l.first << ":" << l.second << std::endl;
        // }

        // Prepare next-level transactions and compute subtree utility.
        std::map<std::vector<int>, Transaction, VectorComparator> nTransactions2;
        std::unordered_map<int, int> subtreeUtil;
        for (const auto &entry : nTransactions) {
            const Transaction &trans = entry.second;
            std::vector<int> filteredKeys;
            std::vector<int> filteredVals;
            for (size_t i = 0; i < trans.keys.size(); ++i) {
                int key = trans.keys[i];
                if (localUtil[key] >= minUtil) {
                    filteredKeys.push_back(key);
                    filteredVals.push_back(trans.vals[i]);
                }
            }
            if (filteredKeys.empty())
                continue;
            int tot = trans.acc;
            for (int v : filteredVals)
                tot += v;
            int temp = 0;
            for (int key : filteredKeys) {
                subtreeUtil[key] += tot - temp;
                // Advance temp by the corresponding utility.
                auto it = std::find(filteredKeys.begin(), filteredKeys.end(), key);
                size_t idx = std::distance(filteredKeys.begin(), it);
                temp += filteredVals[idx];
            }
            // Insert/update in nTransactions2.
            auto it = nTransactions2.find(filteredKeys);
            if (it == nTransactions2.end()) {
                nTransactions2[filteredKeys] = Transaction{filteredKeys, filteredVals, trans.acc};
            } else {
                Transaction &existing = it->second;
                for (size_t i = 0; i < filteredVals.size(); ++i)
                    existing.vals[i] += filteredVals[i];
                existing.acc += trans.acc;
            }
        }

        // // Print subtree utilities for debugging.
        // std::cout << "Subtree Util: " << std::endl;
        // for (const auto &s : subtreeUtil) {
        //     std::cout << s.first << ":" << s.second << std::endl;
        // }
        // std::cout << std::endl;

        // Determine next-level primary items.
        std::vector<int> nextPrimary;
        for (const auto &p : subtreeUtil) {
            if (p.second >= minUtil)
                nextPrimary.push_back(p.first);
        }

        // Recurse if there are any primary items.
        if (!nextPrimary.empty()) {
            std::vector<int> newPrefix(prefix);
            newPrefix.push_back(item);
            _search(newPrefix, nTransactions2, nextPrimary);
        }
    } // end for each candidate item in primary
}

};

/////////////////////
// Example usage //
/////////////////////

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " inputFile minUtil [outputFile]\n";
        return 1;
    }
    std::string inputFile = argv[1];
    int minUtil = std::stoi(argv[2]);
    std::string outFile;
    if (argc > 3)
        outFile = argv[3];

    efim miner(inputFile, minUtil, " ");
    miner.startMine();
    miner.printResults();

    if (!outFile.empty()) {
        miner.save(outFile);
        std::cout << "Patterns saved to: " << outFile << "\n";
    }
    
    return 0;
}
