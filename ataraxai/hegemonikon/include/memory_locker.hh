#pragma once
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>

#include "argon2/argon2.h"
#include "plateform_memory.hh"

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
    size_t capacity;
    bool is_locked;

    void allocate_memory(size_t size)
    {
        size_t page_size = PlatformMemory::get_page_size();
        capacity = (size + page_size - 1) & ~(page_size - 1);

        data = static_cast<char *>(PlatformMemory::allocate_aligned(capacity, page_size));
        if (!data)
        {
            throw std::bad_alloc();
        }

        if (!PlatformMemory::lock_memory(data, capacity))
        {
            PlatformMemory::deallocate_aligned(data);
            throw std::runtime_error("Failed to lock memory: " + PlatformMemory::get_last_error());
        }

        secure_memset(data, 0, capacity);
    }

    void deallocate_memory()
    {
        if (data)
        {
            secure_memset(data, 0, capacity);
            PlatformMemory::unlock_memory(data, capacity);
            PlatformMemory::deallocate_aligned(data);
            data = nullptr;
            capacity = 0;
            length = 0;
        }
    }

public:
    explicit SecureString(const char *str) : data(nullptr), length(0), capacity(0), is_locked(false)
    {
        if (str)
        {
            length = strlen(str);
            allocate_memory(length + 1);
            memcpy(data, str, length + 1);
        }
    }

    explicit SecureString(const std::string &str) : data(nullptr), length(0), capacity(0), is_locked(false)
    {
        if (!str.empty())
        {
            length = str.size();
            allocate_memory(length + 1);
            memcpy(data, str.c_str(), length + 1);
        }
    }

    SecureString() : data(nullptr), length(0), capacity(0), is_locked(false) {}

    ~SecureString()
    {
        deallocate_memory();
    }

    SecureString(const SecureString &) = delete;
    SecureString &operator=(const SecureString &) = delete;

    SecureString(SecureString &&other) noexcept
        : data(other.data), length(other.length), capacity(other.capacity), is_locked(other.is_locked)
    {
        other.data = nullptr;
        other.length = 0;
        other.capacity = 0;
        other.is_locked = false;
    }

    SecureString &operator=(SecureString &&other) noexcept
    {
        if (this != &other)
        {
            deallocate_memory();

            data = other.data;
            length = other.length;
            capacity = other.capacity;
            is_locked = other.is_locked;

            other.data = nullptr;
            other.length = 0;
            other.capacity = 0;
            other.is_locked = false;
        }
        return *this;
    }

    const char *c_str() const { return data ? data : ""; }
    size_t size() const { return length; }
    bool empty() const { return length == 0; }
};

class LockedMemory
{
private:
    void *ptr;
    size_t size;

public:
    /**
     * @brief Constructs a LockedMemory object that allocates and locks a memory region.
     *
     * Allocates a memory block of the specified size with page alignment using PlatformMemory.
     * The allocated memory is then locked into RAM to prevent it from being swapped out.
     * If allocation or locking fails, the constructor throws an exception.
     *
     * @param sz The size (in bytes) of the memory region to allocate and lock.
     * @throws std::bad_alloc if memory allocation fails.
     * @throws std::runtime_error if locking the memory fails.
     */
    LockedMemory(size_t sz) : ptr(nullptr), size(sz)
    {
        size_t page_size = PlatformMemory::get_page_size();
        ptr = PlatformMemory::allocate_aligned(size, page_size);
        if (!ptr)
            throw std::bad_alloc();

        if (!PlatformMemory::lock_memory(ptr, size))
        {
            PlatformMemory::deallocate_aligned(ptr);
            throw std::runtime_error("Failed to lock memory: " + PlatformMemory::get_last_error());
        }
    }

    /**
     * @brief Move constructor for LockedMemory.
     *
     * Transfers ownership of the locked memory from another LockedMemory instance.
     * After the move, the source instance is left in a valid but empty state.
     *
     * @param other The LockedMemory instance to move from.
     */
    LockedMemory(LockedMemory&& other) noexcept
        : ptr(other.ptr), size(other.size)
    {
        other.ptr = nullptr;
        other.size = 0;
    }

    /**
     * @brief Move assignment operator for LockedMemory.
     *
     * Transfers ownership of the locked memory from another LockedMemory instance.
     * If this instance already owns memory, it securely wipes the contents,
     * unlocks the memory, and frees it before taking ownership of the other's memory.
     * The source instance is left in a valid but empty state.
     *
     * @param other The LockedMemory instance to move from.
     * @return Reference to this LockedMemory instance.
     */
    LockedMemory& operator=(LockedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                secure_memset(ptr, 0, size);
                PlatformMemory::unlock_memory(ptr, size);
                PlatformMemory::deallocate_aligned(ptr);
            }

            ptr = other.ptr;
            size = other.size;

            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }

    /**
     * @brief Destructor for the LockedMemory class.
     *
     * This destructor securely erases the memory region pointed to by `ptr` using `secure_memset`,
     * unlocks the memory, and then frees the allocated memory.
     * These steps ensure that sensitive data is not left in memory after the object is destroyed.
     *
     * If `ptr` is null, no action is taken.
     */
    ~LockedMemory()
    {
        if (ptr)
        {
            secure_memset(ptr, 0, size);
            PlatformMemory::unlock_memory(ptr, size);
            PlatformMemory::deallocate_aligned(ptr);
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
    /**
     * @brief Constructs a ProtectedMemory object that allocates and locks a memory region.
     *
     * This constructor allocates a memory region of the specified size using platform-specific
     * memory mapping with read and write permissions. The allocated memory is then locked into 
     * RAM to prevent it from being swapped to disk. If either allocation or locking fails, a
     * std::runtime_error is thrown.
     *
     * @param sz The size (in bytes) of the memory region to allocate and lock.
     * @throws std::runtime_error if memory allocation or locking fails.
     */
    ProtectedMemory(size_t sz) : ptr(nullptr), size(sz)
    {
        ptr = PlatformMemory::allocate_protected(size);
        if (!ptr)
        {
            throw std::runtime_error("Protected memory allocation failed");
        }

        if (!PlatformMemory::lock_memory(ptr, size))
        {
            PlatformMemory::deallocate_protected(ptr, size);
            throw std::runtime_error("Failed to lock protected memory: " + PlatformMemory::get_last_error());
        }
    }

    /**
     * @brief Destructor for the ProtectedMemory class.
     *
     * This destructor securely erases the memory region pointed to by `ptr` by overwriting it with zeros
     * using `secure_memset`, then unlocks the memory and unmaps it.
     * These operations are only performed if `ptr` is not null.
     *
     * Ensures that sensitive data is not left in memory after the object is destroyed.
     */
    ~ProtectedMemory()
    {
        if (ptr)
        {
            secure_memset(ptr, 0, size);
            PlatformMemory::unlock_memory(ptr, size);
            PlatformMemory::deallocate_protected(ptr, size);
        }
    }

    /**
     * @brief Sets the memory region pointed to by `ptr` to read-only.
     *
     * This function changes the protection of the memory region of size `size`
     * starting at address `ptr` to allow only read access.
     * Any attempt to write to this region after this call will result in an access violation.
     *
     * @note The `ptr` and `size` members must be properly initialized before calling this function.
     */
    void make_readonly()
    {
        PlatformMemory::protect_readonly(ptr, size);
    }

    void *get() { return ptr; }

    /**
     * @brief Changes the memory protection of the region pointed to by `ptr` to allow both reading and writing.
     *
     * This function sets the memory region of size `size` starting at `ptr`
     * to have read and write permissions.
     * It is typically used to temporarily make a memory region writable, for example,
     * when modifying code or data that is otherwise protected.
     */
    void make_readwrite()
    {
        PlatformMemory::protect_readwrite(ptr, size);
    }
};

template <typename T>
class SecureAllocator
{
public:
    using value_type = T;

    /**
     * @brief Allocates aligned memory for n objects of type T and locks it into RAM.
     *
     * This function allocates memory aligned to page boundaries for an array of n elements of type T.
     * The allocated memory is then locked into physical RAM to prevent it from being swapped out.
     * If allocation or locking fails, appropriate exceptions are thrown.
     *
     * @param n The number of objects of type T to allocate.
     * @return Pointer to the allocated and locked memory, cast to T*.
     * @throws std::bad_alloc if memory allocation fails.
     * @throws std::runtime_error if locking the memory fails.
     */
    T *allocate(size_t n)
    {
        size_t size = n * sizeof(T);
        size_t page_size = PlatformMemory::get_page_size();
        void *ptr = PlatformMemory::allocate_aligned(size, page_size);
        if (!ptr)
            throw std::bad_alloc();

        if (!PlatformMemory::lock_memory(ptr, size))
        {
            PlatformMemory::deallocate_aligned(ptr);
            throw std::runtime_error("Failed to lock memory: " + PlatformMemory::get_last_error());
        }

        return static_cast<T *>(ptr);
    }

    /**
     * @brief Deallocates memory securely by zeroing out its contents, unlocking it from RAM, and freeing it.
     *
     * This function first overwrites the memory pointed to by `ptr` with zeros using `secure_memset`
     * to prevent sensitive data from lingering in memory. It then unlocks the memory region and
     * finally frees the memory.
     *
     * @tparam T Type of the elements pointed to by `ptr`.
     * @param ptr Pointer to the memory block to deallocate.
     * @param n Number of elements of type `T` to deallocate.
     */
    void deallocate(T *ptr, size_t n)
    {
        if (ptr)
        {
            size_t size = n * sizeof(T);
            secure_memset(ptr, 0, size);
            PlatformMemory::unlock_memory(ptr, size);
            PlatformMemory::deallocate_aligned(ptr);
        }
    }
};

class SecureKey
{
private:
    std::unique_ptr<uint8_t, decltype(&PlatformMemory::deallocate_aligned)> key_data;
    size_t key_size;

public:
    SecureKey(const uint8_t *key, size_t size)
        : key_data(nullptr, &PlatformMemory::deallocate_aligned), key_size(size)
    {
        size_t page_size = PlatformMemory::get_page_size();

        void *ptr = PlatformMemory::allocate_aligned(key_size, page_size);
        if (!ptr)
            throw std::bad_alloc();
        if (!PlatformMemory::lock_memory(ptr, key_size))
        {
            PlatformMemory::deallocate_aligned(ptr);
            throw std::runtime_error("Failed to lock key memory");
        }
        key_data.reset(static_cast<uint8_t *>(ptr));
        memcpy(key_data.get(), key, key_size);
    }

    ~SecureKey()
    {
        if (key_data)
        {
            secure_memset(key_data.get(), 0, key_size);
            PlatformMemory::unlock_memory(key_data.get(), key_size);
        }
    }

    SecureKey(const SecureKey &) = delete;
    SecureKey &operator=(const SecureKey &) = delete;

    SecureKey(SecureKey &&other) noexcept
        : key_data(std::move(other.key_data)), key_size(other.key_size)
    {
        other.key_size = 0;
    }

    SecureKey &operator=(SecureKey &&other) noexcept
    {
        if (this != &other)
        {
            key_data = std::move(other.key_data);
            key_size = other.key_size;
            other.key_size = 0;
        }
        return *this;
    }

    const uint8_t *data() const { return key_data.get(); }
    size_t size() const { return key_size; }
};

using SecureVector = std::vector<uint8_t, SecureAllocator<uint8_t>>;

SecureVector derive_key_from_password(const SecureString &password, const std::vector<uint8_t> &salt)
{
    const uint32_t t_cost = 2;
    const uint32_t m_cost = 65536;
    const uint32_t parallelism = 1;
    const uint32_t key_length = 32;

    SecureVector derived_key(key_length);

    Argon2_Context context(derived_key.data(), static_cast<uint32_t>(derived_key.size()),
                           (uint8_t *)password.c_str(), static_cast<uint32_t>(password.size()),
                           (uint8_t *)salt.data(), static_cast<uint32_t>(salt.size()), nullptr, 0, nullptr, 0,
                           t_cost, m_cost, parallelism, parallelism, nullptr, nullptr, false,
                           false, false, false);

    int result = Argon2id(&context);

    if (result != ARGON2_OK)
    {
        throw std::runtime_error("Argon2 key derivation failed: " + std::string(ErrorMessage(result)));
    }
    return derived_key;
}

SecureKey derive_and_protect_key(const SecureString &password, const std::vector<uint8_t> &salt)
{
    SecureVector key_data = derive_key_from_password(password, salt);

    size_t key_size = key_data.size();
    SecureKey secure_key(key_data.data(), key_size);

    secure_memset(key_data.data(), 0, key_size);

    return secure_key;
}