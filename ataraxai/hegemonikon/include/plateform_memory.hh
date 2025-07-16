#pragma once

#ifdef __linux__
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <cstdlib>
#endif

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#endif

#ifdef __APPLE__
#include <unistd.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <cstdlib>
#endif

#ifdef __FreeBSD__
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <cstdlib>
#endif

#include <cstddef>
#include <stdexcept>
#include <string>

class PlatformMemory {
public:
    enum class Protection {
        NONE = 0,
        READ = 1,
        WRITE = 2,
        EXECUTE = 4,
        READ_WRITE = READ | WRITE,
        READ_EXECUTE = READ | EXECUTE,
        ALL = READ | WRITE | EXECUTE
    };

    static bool lock_memory(void* ptr, size_t size);
    static bool unlock_memory(void* ptr, size_t size);
    static bool protect_memory(void* ptr, size_t size, Protection protection);
    static size_t get_page_size();
    static void* allocate_aligned(size_t size, size_t alignment);
    static void deallocate_aligned(void* ptr);
    static std::string get_last_error();
    static bool is_memory_locked(void* ptr, size_t size);
    static size_t get_memory_usage();
    static bool flush_instruction_cache(void* ptr, size_t size);
    
private:
    static int protection_to_native(Protection protection);
    static thread_local std::string last_error;
};

thread_local std::string PlatformMemory::last_error;

#ifdef __linux__

bool PlatformMemory::lock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (mlock(ptr, size) == 0) {
        return true;
    }
    
    switch (errno) {
        case ENOMEM:
            last_error = "Insufficient memory or process limit exceeded";
            break;
        case EPERM:
            last_error = "Insufficient privileges to lock memory";
            break;
        case EINVAL:
            last_error = "Invalid memory address or size";
            break;
        default:
            last_error = "Unknown error in mlock: " + std::to_string(errno);
    }
    return false;
}

bool PlatformMemory::unlock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (munlock(ptr, size) == 0) {
        return true;
    }
    
    switch (errno) {
        case ENOMEM:
            last_error = "Memory was not locked";
            break;
        case EINVAL:
            last_error = "Invalid memory address or size";
            break;
        default:
            last_error = "Unknown error in munlock: " + std::to_string(errno);
    }
    return false;
}

bool PlatformMemory::protect_memory(void* ptr, size_t size, Protection protection) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    int prot = protection_to_native(protection);
    if (mprotect(ptr, size, prot) == 0) {
        return true;
    }
    
    switch (errno) {
        case EACCES:
            last_error = "Memory protection change not allowed";
            break;
        case EINVAL:
            last_error = "Invalid memory address, size, or protection";
            break;
        case ENOMEM:
            last_error = "Insufficient memory for protection change";
            break;
        default:
            last_error = "Unknown error in mprotect: " + std::to_string(errno);
    }
    return false;
}

size_t PlatformMemory::get_page_size() {
    static size_t page_size = 0;
    if (page_size == 0) {
        page_size = static_cast<size_t>(getpagesize());
    }
    return page_size;
}

void* PlatformMemory::allocate_aligned(size_t size, size_t alignment) {
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        last_error = "Invalid size or alignment (must be power of 2)";
        return nullptr;
    }
    
    void* ptr = aligned_alloc(alignment, size);
    if (!ptr) {
        last_error = "aligned_alloc failed";
    }
    return ptr;
}

void PlatformMemory::deallocate_aligned(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

bool PlatformMemory::is_memory_locked(void* ptr, size_t size) {
    return true; 
}

size_t PlatformMemory::get_memory_usage() {
    return 0; 
}

bool PlatformMemory::flush_instruction_cache(void* ptr, size_t size) {
    __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + size);
    return true;
}

int PlatformMemory::protection_to_native(Protection protection) {
    int prot = 0;
    if (static_cast<int>(protection) & static_cast<int>(Protection::READ)) {
        prot |= PROT_READ;
    }
    if (static_cast<int>(protection) & static_cast<int>(Protection::WRITE)) {
        prot |= PROT_WRITE;
    }
    if (static_cast<int>(protection) & static_cast<int>(Protection::EXECUTE)) {
        prot |= PROT_EXEC;
    }
    if (protection == Protection::NONE) {
        prot = PROT_NONE;
    }
    return prot;
}

#endif // __linux__

#ifdef _WIN32

bool PlatformMemory::lock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (VirtualLock(ptr, size)) {
        return true;
    }
    
    DWORD error = GetLastError();
    switch (error) {
        case ERROR_NOT_ENOUGH_MEMORY:
            last_error = "Insufficient memory to lock pages";
            break;
        case ERROR_INVALID_PARAMETER:
            last_error = "Invalid memory address or size";
            break;
        case ERROR_PRIVILEGE_NOT_HELD:
            last_error = "Insufficient privileges to lock memory";
            break;
        default:
            last_error = "VirtualLock failed with error: " + std::to_string(error);
    }
    return false;
}

bool PlatformMemory::unlock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (VirtualUnlock(ptr, size)) {
        return true;
    }
    
    DWORD error = GetLastError();
    switch (error) {
        case ERROR_NOT_LOCKED:
            last_error = "Memory was not locked";
            break;
        case ERROR_INVALID_PARAMETER:
            last_error = "Invalid memory address or size";
            break;
        default:
            last_error = "VirtualUnlock failed with error: " + std::to_string(error);
    }
    return false;
}

bool PlatformMemory::protect_memory(void* ptr, size_t size, Protection protection) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    DWORD new_protect = protection_to_native(protection);
    DWORD old_protect;
    
    if (VirtualProtect(ptr, size, new_protect, &old_protect)) {
        return true;
    }
    
    DWORD error = GetLastError();
    switch (error) {
        case ERROR_INVALID_PARAMETER:
            last_error = "Invalid memory address, size, or protection";
            break;
        case ERROR_INVALID_ADDRESS:
            last_error = "Invalid memory address";
            break;
        default:
            last_error = "VirtualProtect failed with error: " + std::to_string(error);
    }
    return false;
}

size_t PlatformMemory::get_page_size() {
    static size_t page_size = 0;
    if (page_size == 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        page_size = static_cast<size_t>(si.dwPageSize);
    }
    return page_size;
}

void* PlatformMemory::allocate_aligned(size_t size, size_t alignment) {
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        last_error = "Invalid size or alignment (must be power of 2)";
        return nullptr;
    }
    
    void* ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        last_error = "_aligned_malloc failed";
    }
    return ptr;
}

void PlatformMemory::deallocate_aligned(void* ptr) {
    if (ptr) {
        _aligned_free(ptr);
    }
}

bool PlatformMemory::is_memory_locked(void* ptr, size_t size) {
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(ptr, &mbi, sizeof(mbi)) == sizeof(mbi)) {
        return (mbi.State & MEM_COMMIT) && !(mbi.State & MEM_FREE);
    }
    return false;
}

size_t PlatformMemory::get_memory_usage() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}

bool PlatformMemory::flush_instruction_cache(void* ptr, size_t size) {
    return FlushInstructionCache(GetCurrentProcess(), ptr, size) != FALSE;
}

int PlatformMemory::protection_to_native(Protection protection) {
    switch (protection) {
        case Protection::NONE:
            return PAGE_NOACCESS;
        case Protection::READ:
            return PAGE_READONLY;
        case Protection::WRITE:
            return PAGE_READWRITE; // Windows doesn't have write-only
        case Protection::READ_WRITE:
            return PAGE_READWRITE;
        case Protection::EXECUTE:
            return PAGE_EXECUTE;
        case Protection::READ_EXECUTE:
            return PAGE_EXECUTE_READ;
        case Protection::ALL:
            return PAGE_EXECUTE_READWRITE;
        default:
            return PAGE_NOACCESS;
    }
}

#endif // _WIN32

#ifdef __APPLE__

bool PlatformMemory::lock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (mlock(ptr, size) == 0) {
        return true;
    }
    
    switch (errno) {
        case ENOMEM:
            last_error = "Insufficient memory or process limit exceeded";
            break;
        case EPERM:
            last_error = "Insufficient privileges to lock memory";
            break;
        case EINVAL:
            last_error = "Invalid memory address or size";
            break;
        default:
            last_error = "Unknown error in mlock: " + std::to_string(errno);
    }
    return false;
}

bool PlatformMemory::unlock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (munlock(ptr, size) == 0) {
        return true;
    }
    
    switch (errno) {
        case ENOMEM:
            last_error = "Memory was not locked";
            break;
        case EINVAL:
            last_error = "Invalid memory address or size";
            break;
        default:
            last_error = "Unknown error in munlock: " + std::to_string(errno);
    }
    return false;
}

bool PlatformMemory::protect_memory(void* ptr, size_t size, Protection protection) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    int prot = protection_to_native(protection);
    if (mprotect(ptr, size, prot) == 0) {
        return true;
    }
    
    switch (errno) {
        case EACCES:
            last_error = "Memory protection change not allowed";
            break;
        case EINVAL:
            last_error = "Invalid memory address, size, or protection";
            break;
        case ENOMEM:
            last_error = "Insufficient memory for protection change";
            break;
        default:
            last_error = "Unknown error in mprotect: " + std::to_string(errno);
    }
    return false;
}

size_t PlatformMemory::get_page_size() {
    static size_t page_size = 0;
    if (page_size == 0) {
        page_size = static_cast<size_t>(getpagesize());
    }
    return page_size;
}

void* PlatformMemory::allocate_aligned(size_t size, size_t alignment) {
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        last_error = "Invalid size or alignment (must be power of 2)";
        return nullptr;
    }
    
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    
    last_error = "posix_memalign failed";
    return nullptr;
}

void PlatformMemory::deallocate_aligned(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

bool PlatformMemory::is_memory_locked(void* ptr, size_t size) {
    return true; 
}

size_t PlatformMemory::get_memory_usage() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, 
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
}

bool PlatformMemory::flush_instruction_cache(void* ptr, size_t size) {
    __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + size);
    return true;
}

int PlatformMemory::protection_to_native(Protection protection) {
    int prot = 0;
    if (static_cast<int>(protection) & static_cast<int>(Protection::READ)) {
        prot |= PROT_READ;
    }
    if (static_cast<int>(protection) & static_cast<int>(Protection::WRITE)) {
        prot |= PROT_WRITE;
    }
    if (static_cast<int>(protection) & static_cast<int>(Protection::EXECUTE)) {
        prot |= PROT_EXEC;
    }
    if (protection == Protection::NONE) {
        prot = PROT_NONE;
    }
    return prot;
}

#endif 

#ifdef __FreeBSD__

bool PlatformMemory::lock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (mlock(ptr, size) == 0) {
        return true;
    }
    
    switch (errno) {
        case ENOMEM:
            last_error = "Insufficient memory or process limit exceeded";
            break;
        case EPERM:
            last_error = "Insufficient privileges to lock memory";
            break;
        case EINVAL:
            last_error = "Invalid memory address or size";
            break;
        default:
            last_error = "Unknown error in mlock: " + std::to_string(errno);
    }
    return false;
}

bool PlatformMemory::unlock_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    if (munlock(ptr, size) == 0) {
        return true;
    }
    
    switch (errno) {
        case ENOMEM:
            last_error = "Memory was not locked";
            break;
        case EINVAL:
            last_error = "Invalid memory address or size";
            break;
        default:
            last_error = "Unknown error in munlock: " + std::to_string(errno);
    }
    return false;
}

bool PlatformMemory::protect_memory(void* ptr, size_t size, Protection protection) {
    if (!ptr || size == 0) {
        last_error = "Invalid parameters";
        return false;
    }
    
    int prot = protection_to_native(protection);
    if (mprotect(ptr, size, prot) == 0) {
        return true;
    }
    
    switch (errno) {
        case EACCES:
            last_error = "Memory protection change not allowed";
            break;
        case EINVAL:
            last_error = "Invalid memory address, size, or protection";
            break;
        case ENOMEM:
            last_error = "Insufficient memory for protection change";
            break;
        default:
            last_error = "Unknown error in mprotect: " + std::to_string(errno);
    }
    return false;
}

size_t PlatformMemory::get_page_size() {
    static size_t page_size = 0;
    if (page_size == 0) {
        page_size = static_cast<size_t>(getpagesize());
    }
    return page_size;
}

void* PlatformMemory::allocate_aligned(size_t size, size_t alignment) {
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        last_error = "Invalid size or alignment (must be power of 2)";
        return nullptr;
    }
    
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    
    last_error = "posix_memalign failed";
    return nullptr;
}

void PlatformMemory::deallocate_aligned(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

bool PlatformMemory::is_memory_locked(void* ptr, size_t size) {
    return true; 
}

size_t PlatformMemory::get_memory_usage() {
    return 0;
}

bool PlatformMemory::flush_instruction_cache(void* ptr, size_t size) {
    __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + size);
    return true;
}

int PlatformMemory::protection_to_native(Protection protection) {
    int prot = 0;
    if (static_cast<int>(protection) & static_cast<int>(Protection::READ)) {
        prot |= PROT_READ;
    }
    if (static_cast<int>(protection) & static_cast<int>(Protection::WRITE)) {
        prot |= PROT_WRITE;
    }
    if (static_cast<int>(protection) & static_cast<int>(Protection::EXECUTE)) {
        prot |= PROT_EXEC;
    }
    if (protection == Protection::NONE) {
        prot = PROT_NONE;
    }
    return prot;
}

#endif

std::string PlatformMemory::get_last_error() {
    return last_error;
}

namespace PlatformMemoryUtils {
    bool is_platform_supported() {
        #if defined(__linux__) || defined(_WIN32) || defined(__APPLE__) || defined(__FreeBSD__)
            return true;
        #else
            return false;
        #endif
    }
    
    std::string get_platform_name() {
        #ifdef __linux__
            return "Linux";
        #elif _WIN32
            return "Windows";
        #elif __APPLE__
            return "macOS";
        #elif __FreeBSD__
            return "FreeBSD";
        #else
            return "Unknown";
        #endif
    }
    
    size_t align_size(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    bool is_power_of_two(size_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }
}