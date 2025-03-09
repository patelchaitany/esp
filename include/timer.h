#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <functional>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string label;

public:
    Timer(const std::string& label = "") : label(label) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        stop();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double ms = duration.count() / 1000.0;
        
        if (!label.empty()) {
            std::cout << label << ": " << ms << " ms" << std::endl;
        }
        
        return ms;
    }

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    static double measureBlock(const std::string& label, const std::function<void()>& block) {
        Timer timer(label);
        block();
        return timer.stop();
    }
};