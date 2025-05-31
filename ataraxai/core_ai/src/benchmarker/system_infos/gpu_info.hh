#include <string>
#include <vector>

#include <cstdlib>
#include <cstdio>
#include <iostream>

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

struct GPUInfo
{
    std::string name;
    std::string vendor;
    std::string driver_version;
    size_t memory_total_MB;
    std::string compute_capability;

    GPUInfo()
        : name("Unknown"), vendor("Unknown"), driver_version("Unknown"), memory_total_MB(0), compute_capability("Unknown")
    {
    }
};

std::ostream &operator<<(std::ostream &os, const GPUInfo &info)
{
    os << "GPU Name: " << info.name << "\n"
       << "  Vendor: " << info.vendor << "\n"
       << "  Driver Version: " << info.driver_version << "\n"
       << "  Memory Total: " << info.memory_total_MB << " MB\n"
       << "  Compute Capability: " << info.compute_capability;
    return os;
}

struct GPUInfoCollection
{
    std::vector<GPUInfo> gpus;

    void set_linux_gpu_info()
    {
        FILE *pipe = popen("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits", "r");
        if (!pipe)
            return;

        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe))
        {
            GPUInfo info;
            std::string line(buffer);
            size_t pos1 = line.find(",");
            size_t pos2 = line.find(",", pos1 + 1);

            if (pos1 != std::string::npos && pos2 != std::string::npos)
            {
                info.name = line.substr(0, pos1);
                info.memory_total_MB = std::stoul(line.substr(pos1 + 1, pos2 - pos1 - 1));
                info.driver_version = line.substr(pos2 + 1);
                info.vendor = "NVIDIA";
            }
            gpus.push_back(info);
        }
        pclose(pipe);
    }

    void set_windows_gpu_info()
    {
#ifdef _WIN32
        // Use DXGI or WMI to query GPU info, e.g., via `wmic` command for simplicity
        FILE *pipe = _popen("wmic path win32_VideoController get Name,AdapterRAM,DriverVersion /format:csv", "r");
        if (!pipe)
            return;

        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe))
        {
            std::string line(buffer);
            if (line.find("Name") != std::string::npos)
                continue;
            std::istringstream iss(line);
            std::string node, name, ram_str, driver;
            std::getline(iss, node, ','); // node name
            std::getline(iss, name, ',');
            std::getline(iss, ram_str, ',');
            std::getline(iss, driver, ',');

            gpu_info info;
            info.name = name;
            info.memory_total_MB = std::stoul(ram_str) / (1024 * 1024);
            info.driver_version = driver;
            info.vendor = "Unknown";
            gpus.push_back(info);
        }
        _pclose(pipe);
#endif
    }

    void set_macos_gpu_info()
    {
#ifdef __APPLE__
        // Use system_profiler to get GPU info
        FILE *pipe = popen("system_profiler SPDisplaysDataType | grep 'Chipset Model\\|VRAM\\|Driver Version'", "r");
        if (!pipe)
            return;

        char buffer[256];
        gpu_info info;
        while (fgets(buffer, sizeof(buffer), pipe))
        {
            std::string line(buffer);
            if (line.find("Chipset Model") != std::string::npos)
            {
                info.name = line.substr(line.find(":") + 2);
            }
            else if (line.find("VRAM") != std::string::npos)
            {
                info.memory_total_MB = std::stoul(line.substr(line.find(":") + 2)) / (1024 * 1024);
            }
            else if (line.find("Driver Version") != std::string::npos)
            {
                info.driver_version = line.substr(line.find(":") + 2);
                info.vendor = "Apple";
                gpus.push_back(info);
            }
        }
        pclose(pipe);
#endif
    }

    GPUInfoCollection()
    {
#ifdef __linux__
        set_linux_gpu_info();
#elif _WIN32
        set_windows_gpu_info();
#elif __APPLE__
        set_macos_gpu_info();
#endif
    }
};
