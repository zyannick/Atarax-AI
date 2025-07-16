
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>
#include "memory_locker.hh"

TEST_CASE("Test secure memset", "[service][unit]") {
    char buffer[16];
    memset(buffer, 'A', sizeof(buffer));
    secure_memset(buffer, 0x5A, sizeof(buffer));
    for (size_t i = 0; i < sizeof(buffer); ++i) {
        REQUIRE(buffer[i] == 0x5A);
    }
}

TEST_CASE("Test SecureString", "[service][unit]") {
    const char *test_str = "secret";
    SecureString s(test_str);
    REQUIRE(strcmp(s.c_str(), test_str) == 0);
    REQUIRE(s.size() == strlen(test_str));
    REQUIRE(!s.empty());

    SecureString empty;
    REQUIRE(empty.size() == 0);
    REQUIRE(empty.empty());
    REQUIRE(strcmp(empty.c_str(), "") == 0);
}

TEST_CASE("Test LockedMemory", "[service][unit]") {
    LockedMemory mem(32);
    void *ptr = mem.get();
    REQUIRE(ptr != nullptr);
    memset(ptr, 0xAB, 32);
}

TEST_CASE("Test ProtectedMemory", "[service][unit]") {
    ProtectedMemory mem(64);
    void *ptr = mem.get();
    REQUIRE(ptr != nullptr);
    memset(ptr, 0xCD, 64);
    mem.make_readonly();
    mem.make_readwrite();
}

TEST_CASE("Test SecureAllocator", "[service][unit]") {
    SecureAllocator<uint8_t> alloc;
    uint8_t *data = alloc.allocate(10);
    REQUIRE(data != nullptr);
    memset(data, 0xEF, 10);
    alloc.deallocate(data, 10);
}

TEST_CASE("Test SecureKey", "[service][unit]") {
    std::vector<uint8_t> key(32, 0x11);
    SecureKey k(key.data(), key.size());
    REQUIRE(k.size() == key.size());
    REQUIRE(memcmp(k.data(), key.data(), key.size()) == 0);

    SecureKey k2(std::move(k));
    REQUIRE(k2.size() == key.size());
    REQUIRE(k.size() == 0);
}

TEST_CASE("Test Derive Key from password", "[service][unit]") {
    SecureString password("testpassword");
    std::vector<uint8_t> salt(16, 0x22);
    SecureVector key = derive_key_from_password(password, salt);
    REQUIRE(key.size() == 32);
}

TEST_CASE("Test Derive Key and protect key", "[service][unit]") {
    SecureString password("anotherpassword");
    std::vector<uint8_t> salt(16, 0x33);
    SecureKey key = derive_and_protect_key(password, salt);
    REQUIRE(key.size() == 32);
}


TEST_CASE("Test SecureString move", "[service][unit]") {
    SecureString s1("move_me");
    SecureString s2(std::move(s1));
    REQUIRE(strcmp(s2.c_str(), "move_me") == 0);
    REQUIRE(s1.size() == 0);
    REQUIRE(s1.empty());
    SecureString s3;
    s3 = std::move(s2);
    REQUIRE(strcmp(s3.c_str(), "move_me") == 0);
    REQUIRE(s2.size() == 0);
}

TEST_CASE("Test LockMemory move", "[service][unit]") {
    LockedMemory m1(16);
    void *ptr1 = m1.get();
    LockedMemory m2(std::move(m1));
    REQUIRE(m2.get() == ptr1);
}



TEST_CASE("Test SecureVector", "[service][unit]") {
    SecureVector v(8, 0x77);
    for (auto b : v) REQUIRE(b == 0x77);
    v.resize(16, 0x88);
    for (size_t i = 8; i < v.size(); ++i) REQUIRE(v[i] == 0x88);
}

TEST_CASE("Test SecureKey move assign", "[service][unit]") {
    std::vector<uint8_t> key(16, 0x44);
    SecureKey k1(key.data(), key.size());
    SecureKey k2(std::move(k1));
    SecureKey k3(std::vector<uint8_t>(16, 0x55).data(), 16);
    k3 = std::move(k2);
    REQUIRE(k3.size() == 16);
    REQUIRE(k2.size() == 0);
}
