// args_parser.cu

#include "args.h"
#include <iostream>
#include <sstream>

// Helper function to map separator string to actual character
static char getSeparator(const std::string& sepStr) {
    if (sepStr == "\\s") return ' ';
    if (sepStr == "\\t") return '\t';
    if (sepStr == "\\n") return '\n';
    if (sepStr == ",") return ',';
    if (sepStr == ";") return ';';
    // Add more separators as needed
    // Default separator
    std::cerr << "Unknown separator: " << sepStr << ". Using space as default.\n";
    return ' ';
}

// Function to parse command-line arguments
bool parseArguments(int argc, char* argv[], ParsedArgs& args) {
    // Expected usage: <program> <file> <utility(int)> <separator>
    if (argc != 4) {
        // std::cerr << "Usage: " << argv[0] << " <file> <utility(int)> <separator> <blocks> <threads>\n";
        std::cerr << "Usage: " << argv[0] << " <file> <utility(int)> <separator>\n";
        std::cerr << "Example separators: \\s for space, \\t for tab, \\n for newline, , for comma, ; for semicolon\n";
        return false;
    }

    // Parse filename
    args.filename = argv[1];

    // Parse utility integer
    try {
        args.utility = std::stoi(argv[2]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid utility argument. It must be an integer.\n";
        return false;
    } catch (const std::out_of_range& e) {
        std::cerr << "Utility argument out of range.\n";
        return false;
    }

    // Parse separator
    std::string sepStr = argv[3];
    args.separator = getSeparator(sepStr);

    // // Parse blocks
    // try {
    //     args.blocks = std::stoi(argv[4]);
    // } catch (const std::invalid_argument& e) {
    //     std::cerr << "Invalid blocks argument. It must be an integer.\n";
    //     return false;
    // } catch (const std::out_of_range& e) {
    //     std::cerr << "Blocks argument out of range.\n";
    //     return false;
    // }

    // // Parse threads
    // try {
    //     args.threads = std::stoi(argv[5]);
    // } catch (const std::invalid_argument& e) {
    //     std::cerr << "Invalid threads argument. It must be an integer.\n";
    //     return false;
    // } catch (const std::out_of_range& e) {
    //     std::cerr << "Threads argument out of range.\n";
    //     return false;
    // }

    return true;
}
