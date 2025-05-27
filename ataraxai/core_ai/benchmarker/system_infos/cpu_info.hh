#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <iostream>

#ifdef __linux__
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <winreg.h>
#endif

struct cpu_info
{
    std::string cpu_model;
    int num_cores = 0;
    int num_threads = 0;
    float cpu_frequency = 0.0f;
    std::string architecture;
    std::string cache_size;
    std::string flags;

    cpu_info()
        : cpu_model("Unknown"), num_cores(0), num_threads(0), cpu_frequency(0.0f),
          architecture("Unknown"), cache_size("Unknown"), flags("Unknown") {}
};

struct cpu_info_collection
{
    std::vector<cpu_info> cpus;

    static std::string trim(const std::string &s)
    {
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }

    void set_linux_cpu_infos()
    {
        #ifdef __linux__
            FILE *pipe = popen("lscpu", "r");
            if (!pipe)
                return;

            char buffer[256];
            cpu_info info;
            while (fgets(buffer, sizeof(buffer), pipe))
            {
                std::string line(buffer);
                if (line.find("Model name:") != std::string::npos)
                {
                    info.cpu_model = trim(line.substr(line.find(":") + 1));
                }
                else if (line.find("Thread(s) per core:") != std::string::npos)
                {
                    info.num_threads = std::stoi(trim(line.substr(line.find(":") + 1)));
                }
                else if (line.find("Core(s) per socket:") != std::string::npos)
                {
                    int cores_per_socket = std::stoi(trim(line.substr(line.find(":") + 1)));
                    info.num_cores = cores_per_socket; // will adjust later if multiple sockets
                }
                else if (line.find("Socket(s):") != std::string::npos)
                {
                    int sockets = std::stoi(trim(line.substr(line.find(":") + 1)));
                    info.num_cores *= sockets;
                }
                else if (line.find("CPU MHz:") != std::string::npos)
                {
                    info.cpu_frequency = std::stof(trim(line.substr(line.find(":") + 1)));
                }
                else if (line.find("Architecture:") != std::string::npos)
                {
                    info.architecture = trim(line.substr(line.find(":") + 1));
                }
                else if (line.find("L1d cache:") != std::string::npos)
                {
                    info.cache_size = trim(line.substr(line.find(":") + 1));
                }
                else if (line.find("Flags:") != std::string::npos)
                {
                    info.flags = trim(line.substr(line.find(":") + 1));
                    cpus.push_back(info);
                    info = cpu_info();
                }
            }

            pclose(pipe);
        #endif
    }

    void set_windows_cpu_info()
    {
#ifdef _WIN32
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);

        cpu_info info;
        info.num_cores = sysInfo.dwNumberOfProcessors;

        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                         TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                         0, KEY_READ, &hKey) == ERROR_SUCCESS)
        {
            TCHAR buffer[256];
            DWORD bufferSize = sizeof(buffer);
            if (RegQueryValueEx(hKey, TEXT("ProcessorNameString"), NULL, NULL,
                                (LPBYTE)buffer, &bufferSize) == ERROR_SUCCESS)
            {
                info.cpu_model = buffer;
            }
            RegCloseKey(hKey);
        }

        cpus.push_back(info);
#endif
    }

    void set_macos_cpu_info()
    {
#ifdef __APPLE__
        cpu_info info;

        // CPU model
        char model[256];
        size_t size = sizeof(model);
        sysctlbyname("machdep.cpu.brand_string", model, &size, NULL, 0);
        info.cpu_model = model;

        // Architecture
        char arch[256];
        size = sizeof(arch);
        sysctlbyname("hw.machine", arch, &size, NULL, 0);
        info.architecture = arch;

        // Core count
        int core_count = 0;
        size = sizeof(core_count);
        sysctlbyname("machdep.cpu.core_count", &core_count, &size, NULL, 0);
        info.num_cores = core_count;

        // Logical processors
        int logical_per_package = 0;
        size = sizeof(logical_per_package);
        sysctlbyname("machdep.cpu.logical_per_package", &logical_per_package, &size, NULL, 0);
        info.num_threads = logical_per_package;

        // Frequency
        int freq = 0;
        size = sizeof(freq);
        sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0);
        info.cpu_frequency = static_cast<float>(freq) / 1e6f;

        // Cache
        int cache = 0;
        size = sizeof(cache);
        sysctlbyname("hw.l2cachesize", &cache, &size, NULL, 0);
        info.cache_size = std::to_string(cache / 1024) + " KB";

        // Flags
        char features[1024];
        size = sizeof(features);
        sysctlbyname("machdep.cpu.features", features, &size, NULL, 0);
        info.flags = features;

        cpus.push_back(info);
#endif
    }

    cpu_info_collection()
    {
#ifdef __linux__
        set_linux_cpu_infos();
#elif _WIN32
        set_windows_cpu_info();
#elif __APPLE__
        set_macos_cpu_info();
#endif
    }
};
