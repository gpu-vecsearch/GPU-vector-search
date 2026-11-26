#pragma once

#include <chrono>

// Wall-clock timer (steady_clock). search() is synchronous for every version
// (results are copied back to the host before it returns), so no explicit
// device synchronization is needed around it.
class Timer {
public:
    void start();
    double elapsed_ms() const;

private:
    std::chrono::steady_clock::time_point start_{};
};
