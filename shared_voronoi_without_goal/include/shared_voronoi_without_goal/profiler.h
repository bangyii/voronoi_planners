#ifndef PROFILER_H_
#define PROFILER_H_

#include <string>
#include <chrono>
#include <iostream>

class Profiler
{
public:
    Profiler()
    {
        start = std::chrono::system_clock::now();
    }

    void print(const std::string &output_text_)
    {
        std::cout << output_text_ << ": " << (std::chrono::system_clock::now() - start).count() / 1000000000.0 << "\n";
        start = std::chrono::system_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start;
};

#endif