#pragma once
#include <chrono>
#include <cmath>
#include <time.h>

namespace stsrl::infra {
inline long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double thread_cpu_seconds() {
    timespec t{};
    return ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0 ? t.tv_sec + t.tv_nsec / 1e9 : std::nan("");
}
}  // namespace stsrl::infra
