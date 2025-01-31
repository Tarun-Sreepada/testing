// args.h

#ifndef args_H
#define args_H

#include <string>

// Structure to hold parsed arguments
struct ParsedArgs {
    std::string filename;
    int utility;
    std::string separator;
};

// Function to parse command-line arguments
// Returns true if parsing is successful, false otherwise
bool parseArguments(int argc, char* argv[], ParsedArgs& args);

#endif // args_H
