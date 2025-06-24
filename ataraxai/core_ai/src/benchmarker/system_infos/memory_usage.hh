#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>


using namespace std::chrono;

#include <thread>
unsigned int num_cores = std::thread::hardware_concurrency();


#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h> 



long get_memory_usage_linux()
{
    std::ifstream status_file("/proc/self/status");
    std::string line;
    long rss = 0;
    while (std::getline(status_file, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
        {                                           
            std::istringstream iss(line.substr(6)); 
            iss >> rss;                           
            break;
        }
    }
    return rss * 1024; 
}
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <filesystem>

long get_memory_usage_macos()
{
    mach_task_basic_info_data_t task_info_data;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&task_info_data, &count) == KERN_SUCCESS)
    {
        return task_info_data.resident_size;
    }
    return 0;
}
#endif

long get_current_memory_usage()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc));
    return pmc.PrivateUsage;
#elif defined(__linux__)
    return get_memory_usage_linux();
#elif defined(__APPLE__)
    return get_memory_usage_macos();
#else
    return 0;
#endif
}