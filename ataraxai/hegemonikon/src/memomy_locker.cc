#include <cstring>
#include <memory>
#include <sys/mman.h>
#include <stdexcept>
#include <vector>
#include <iostream>

/**
 * @brief Securely sets a block of memory to a specified value.
 *
 * This function fills the memory area pointed to by `ptr` with the constant byte `value`,
 * for `size` bytes. It uses a volatile pointer to prevent the compiler from optimizing
 * out the memory set operation, which is important for securely erasing sensitive data.
 *
 * @param ptr   Pointer to the memory area to be set.
 * @param value Value to set (interpreted as an unsigned char).
 * @param size  Number of bytes to set.
 */
void secure_memset(void *ptr, int value, size_t size)
{
    volatile char *p = static_cast<volatile char *>(ptr);
    while (size--)
    {
        *p++ = value;
    }
}

class SecureString
{
private:
    char *data;
    size_t length;

public:
    SecureString(const char *str) : length(strlen(str))
    {
        data = new char[length + 1];
        memcpy(data, str, length + 1);
    }

    ~SecureString()
    {
        if (data)
        {
            secure_memset(data, 0, length);
            delete[] data;
        }
    }

    SecureString(const SecureString &) = delete;
    SecureString &operator=(const SecureString &) = delete;
};

class LockedMemory
{
private:
    void *ptr;
    size_t size;

public:
    LockedMemory(size_t sz) : size(sz)
    {
        ptr = aligned_alloc(4096, size); 
        if (!ptr)
            throw std::bad_alloc();

        if (mlock(ptr, size) != 0)
        {
            free(ptr);
            throw std::runtime_error("Failed to lock memory");
        }
    }

    ~LockedMemory()
    {
        if (ptr)
        {
            secure_memset(ptr, 0, size);
            munlock(ptr, size);
            free(ptr);
        }
    }

    void *get() { return ptr; }
};

class ProtectedMemory
{
private:
    void *ptr;
    size_t size;

public:
    ProtectedMemory(size_t sz) : size(sz)
    {
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED)
        {
            throw std::runtime_error("mmap failed");
        }

        if (mlock(ptr, size) != 0)
        {
            munmap(ptr, size);
            throw std::runtime_error("mlock failed");
        }
    }

    ~ProtectedMemory()
    {
        if (ptr != MAP_FAILED)
        {
            secure_memset(ptr, 0, size);
            munlock(ptr, size);
            munmap(ptr, size);
        }
    }

    void make_readonly()
    {
        mprotect(ptr, size, PROT_READ);
    }

    void make_readwrite()
    {
        mprotect(ptr, size, PROT_READ | PROT_WRITE);
    }
};

template <typename T>
class SecureAllocator
{
public:
    using value_type = T;

    T *allocate(size_t n)
    {
        size_t size = n * sizeof(T);
        void *ptr = aligned_alloc(4096, size);
        if (!ptr)
            throw std::bad_alloc();

        if (mlock(ptr, size) != 0)
        {
            free(ptr);
            throw std::runtime_error("Failed to lock memory");
        }

        return static_cast<T *>(ptr);
    }

    void deallocate(T *ptr, size_t n)
    {
        if (ptr)
        {
            size_t size = n * sizeof(T);
            secure_memset(ptr, 0, size);
            munlock(ptr, size);
            free(ptr);
        }
    }
};

// Usage with STL containers
using SecureString = std::basic_string<char, std::char_traits<char>, SecureAllocator<char>>;
using SecureVector = std::vector<uint8_t, SecureAllocator<uint8_t>>;

class SecureKey
{
private:
    std::unique_ptr<uint8_t[]> key_data;
    size_t key_size;
    bool is_locked;

    void lock_memory()
    {
        if (mlock(key_data.get(), key_size) == 0)
        {
            is_locked = true;
        }
    }

    void unlock_memory()
    {
        if (is_locked)
        {
            munlock(key_data.get(), key_size);
            is_locked = false;
        }
    }

public:
    SecureKey(const uint8_t *key, size_t size)
        : key_size(size), is_locked(false)
    {

        // Allocate page-aligned memory
        key_data = std::make_unique<uint8_t[]>(key_size);
        memcpy(key_data.get(), key, key_size);

        lock_memory();
    }

    ~SecureKey()
    {
        if (key_data)
        {
            secure_memset(key_data.get(), 0, key_size);
            unlock_memory();
        }
    }

    // Prevent copying
    SecureKey(const SecureKey &) = delete;
    SecureKey &operator=(const SecureKey &) = delete;

    // Allow moving
    SecureKey(SecureKey &&other) noexcept
        : key_data(std::move(other.key_data)),
          key_size(other.key_size),
          is_locked(other.is_locked)
    {
        other.key_size = 0;
        other.is_locked = false;
    }

    const uint8_t *data() const { return key_data.get(); }
    size_t size() const { return key_size; }
};

class GuardedMemory
{
private:
    void *base_ptr;
    void *data_ptr;
    size_t total_size;
    size_t data_size;

public:
    GuardedMemory(size_t size) : data_size(size)
    {
        size_t page_size = getpagesize();
        total_size = ((size + page_size - 1) / page_size + 2) * page_size;

        base_ptr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (base_ptr == MAP_FAILED)
        {
            throw std::runtime_error("mmap failed");
        }

        // Set guard pages
        mprotect(base_ptr, page_size, PROT_NONE);
        mprotect(static_cast<char *>(base_ptr) + total_size - page_size,
                 page_size, PROT_NONE);

        data_ptr = static_cast<char *>(base_ptr) + page_size;
        mlock(data_ptr, data_size);
    }

    ~GuardedMemory()
    {
        if (base_ptr != MAP_FAILED)
        {
            secure_memset(data_ptr, 0, data_size);
            munlock(data_ptr, data_size);
            munmap(base_ptr, total_size);
        }
    }

    void *get() { return data_ptr; }
};