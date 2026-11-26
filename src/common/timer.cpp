#include "timer.h"

void Timer::start() {
    start_ = std::chrono::steady_clock::now();
}

double Timer::elapsed_ms() const {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
}
