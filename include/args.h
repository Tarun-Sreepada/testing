// args.h

#ifndef args_H
#define args_H

#include <string>
#include <vector>
#include <iostream>
#include <chrono>


// Structure to hold parsed arguments
struct ParsedArgs {
    std::string filename;
    int utility;
    std::string separator;
    int blocks;
    int threads;
};

// Function to parse command-line arguments
// Returns true if parsing is successful, false otherwise
bool parseArguments(int argc, char* argv[], ParsedArgs& args);


struct TimeRecord {
    std::string label;
    std::chrono::high_resolution_clock::time_point timestamp;
};

class Timer {
public:
    // Record a new time point with the provided label.
    void recordPoint(const std::string& label) {
        TimeRecord rec;
        rec.label = label;
        rec.timestamp = std::chrono::high_resolution_clock::now();
        records.push_back(rec);
    }

    // Print the recorded points.
    // The first record is printed as the base (0 ms),
    // then each subsequent record shows the difference from the previous one.
    // Finally, prints the total time between the first and the last record.
    void printRecords() const {
        if (records.empty()) {
            std::cout << "No records to display." << std::endl;
            return;
        }

        // Print the first record as base
        std::cout << records[0].label << ": " << 0 << " ms" << std::endl;
        for (size_t i = 1; i < records.size(); ++i) {
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                records[i].timestamp - records[i-1].timestamp
            ).count();
            std::cout << records[i].label << ": " << diff << " ms" << std::endl;
        }

        // Compute total time from first to last record.
        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
            records.back().timestamp - records.front().timestamp
        ).count();
        std::cout << "Total time: " << total << " ms" << std::endl;
    }

private:
    std::vector<TimeRecord> records;
};


#endif // args_H
