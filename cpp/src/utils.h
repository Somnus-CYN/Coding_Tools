// Small utilities for logging and timing.
#pragma once
#include <chrono>
#include <cstdio>

#define LOGE(fmt, ...) std::fprintf(stderr, "[YOLO11N][E] " fmt "\n", ##__VA_ARGS__)
#define LOGI(fmt, ...) std::fprintf(stdout, "[YOLO11N][I] " fmt "\n", ##__VA_ARGS__)

class ScopedTimer {
public:
    explicit ScopedTimer(const char* name) : name_(name), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        LOGI("%s took %ld ms", name_, static_cast<long>(ms));
    }

private:
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};
