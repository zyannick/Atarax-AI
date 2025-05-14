#include <string>
#include <fstream>
#include <sstream>
#include <thread>

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
    int num_cores;
    int num_threads;
    float cpu_frequency; 
    std::string architecture;
    std::string cache_size; 
    std::string flags;     

    void set_linux_cpu_info()
    {
        std::ifstream cpuinfo_file("/proc/cpuinfo");
        if (cpuinfo_file.is_open())
        {
            std::string line;
            while (std::getline(cpuinfo_file, line))
            {
                if (line.find("model name") != std::string::npos)
                {
                    cpu_model = line.substr(line.find(":") + 2);
                }
                else if (line.find("cpu cores") != std::string::npos)
                {
                    num_cores = std::stoi(line.substr(line.find(":") + 2));
                }
                else if (line.find("siblings") != std::string::npos)
                {
                    num_threads = std::stoi(line.substr(line.find(":") + 2));
                }
                else if (line.find("cpu MHz") != std::string::npos)
                {
                    cpu_frequency = std::stof(line.substr(line.find(":") + 2)) / 1000.0; 
                }
                else if (line.find("flags") != std::string::npos)
                {
                    flags = line.substr(line.find(":") + 2);
                }
            }
            cpuinfo_file.close();
        }

        std::ifstream cacheinfo_file("/proc/cpuinfo");
        if (cacheinfo_file.is_open())
        {
            std::string line;
            while (std::getline(cacheinfo_file, line))
            {
                if (line.find("cache size") != std::string::npos)
                {
                    cache_size = line.substr(line.find(":") + 2);
                    break;
                }
            }
            cacheinfo_file.close();
        }

        std::ifstream version_file("/proc/version");
        if (version_file.is_open())
        {
            std::string line;
            while (std::getline(version_file, line))
            {
                if (line.find("x86_64") != std::string::npos)
                {
                    architecture = "x86_64";
                    break;
                }
                else if (line.find("i686") != std::string::npos)
                {
                    architecture = "i686";
                    break;
                }
            }
            version_file.close();
        }
    }

    void set_windows_cpu_info()
    {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        this->num_cores = sys_info.dwNumberOfProcessors;

        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"), 0, KEY_READ, &hKey) == ERROR_SUCCESS)
        {
            char cpu_model[256];
            DWORD size = sizeof(cpu_model);
            RegQueryValueEx(hKey, TEXT("ProcessorNameString"), NULL, NULL, (LPBYTE)cpu_model, &size);
            this->cpu_model = std::string(cpu_model);

            DWORD frequency;
            size = sizeof(frequency);
            RegQueryValueEx(hKey, TEXT("~MHz"), NULL, NULL, (LPBYTE)&frequency, &size);
            this->cpu_frequency = frequency / 1000.0; // convert to GHz

            RegCloseKey(hKey);
        }

        this->num_threads = std::thread::hardware_concurrency();
    }

    void set_macos_cpu_info()
    {
        // Use sysctl to get CPU information
        int num_cores;
        size_t size = sizeof(num_cores);
        sysctlbyname("hw.physicalcpu", &num_cores, &size, NULL, 0);
        this->num_cores = num_cores;

        int num_threads;
        size = sizeof(num_threads);
        sysctlbyname("hw.logicalcpu", &num_threads, &size, NULL, 0);
        this->num_threads = num_threads;

        char cpu_model[256];
        size = sizeof(cpu_model);
        sysctlbyname("machdep.cpu.brand_string", cpu_model, &size, NULL, 0);
        this->cpu_model = std::string(cpu_model);

        float cpu_frequency;
        size = sizeof(cpu_frequency);
        sysctlbyname("hw.cpufrequency", &cpu_frequency, &size, NULL, 0);
        this->cpu_frequency = cpu_frequency / 1e9; // convert to GHz

        this->architecture = "x86_64"; // macOS is always x86_64
    }


    cpu_info() : cpu_model("Unknown"), num_cores(0), num_threads(0),
             cpu_frequency(0.0f), architecture("Unknown"),
             cache_size("Unknown"), flags("Unknown")
    {
        #ifdef __linux__
            set_linux_cpu_info();
        #elif _WIN32
            set_windows_cpu_info();
        #elif __APPLE__
            set_macos_cpu_info();
        #endif

    }



};