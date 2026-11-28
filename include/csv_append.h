#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

// Append one CSV row, writing `header` first if the file is missing or empty
// and starting the row on a new line if the file lacks a trailing newline.
inline void append_csv_row(const std::string& path, const std::string& header, const std::string& row) {
    bool empty = true;
    bool ends_with_newline = true;
    {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (in && in.tellg() > 0) {
            empty = false;
            in.seekg(-1, std::ios::end);
            char last = '\0';
            in.get(last);
            ends_with_newline = (last == '\n');
        }
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("cannot open " + path + " for appending");
    }
    if (empty) out << header << '\n';
    else if (!ends_with_newline) out << '\n';
    out << row << '\n';
}
