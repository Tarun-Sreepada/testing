// parser.h

#ifndef parser_h
#define parser_h

#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <cstdint>

// Structure to hold the result of the read_file function
struct ReadFileResult {
    std::map<std::vector<int>, std::vector<int>> filteredTransactions;
    std::vector<uint32_t> primary;
    std::unordered_map<int, std::string> rename;
    int max_item;
};

// Function to split a string based on a delimiter
std::vector<std::string> split(const std::string& s, const std::string& delimiter);

// Function to read and parse the input file
ReadFileResult read_file(const std::string& inputFile, const std::string& sep, int minUtil);

#endif // parser_h
