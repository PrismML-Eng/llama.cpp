#include "llama-mmap.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static std::array<bool, 4> resident_pages(void * base, size_t page_size) {
    std::array<PSAPI_WORKING_SET_EX_INFORMATION, 4> entries{};
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i].VirtualAddress = static_cast<char *>(base) + i * page_size;
    }
    require(QueryWorkingSetEx(GetCurrentProcess(), entries.data(), sizeof(entries)) != 0,
            "QueryWorkingSetEx failed");
    std::array<bool, 4> resident{};
    for (size_t i = 0; i < entries.size(); ++i) {
        resident[i] = entries[i].VirtualAttributes.Valid != 0;
    }
    return resident;
}

int main() {
    try {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        const size_t page_size = info.dwPageSize;
        std::vector<unsigned char> expected(8 * page_size);
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<unsigned char>((i / page_size) * 17 + i % 251);
        }
        wchar_t directory[MAX_PATH], filename[MAX_PATH];
        const DWORD path_size = GetTempPathW(MAX_PATH, directory);
        require(path_size > 0 && path_size < MAX_PATH, "GetTempPathW failed");
        require(GetTempFileNameW(directory, L"llm", 0, filename) != 0, "GetTempFileNameW failed");
        FILE * raw = nullptr;
        if (_wfopen_s(&raw, filename, L"w+b") != 0) {
            DeleteFileW(filename);
            throw std::runtime_error("opening fixture failed");
        }
        const auto close_temp = [&](FILE * fp) {
            std::fclose(fp);
            DeleteFileW(filename);
        };
        std::unique_ptr<FILE, decltype(close_temp)> temp(raw, close_temp);
        require(std::fwrite(expected.data(), 1, expected.size(), temp.get()) == expected.size(),
                "writing fixture failed");
        require(std::fflush(temp.get()) == 0, "flushing fixture failed");
        llama_file file(temp.get());
        llama_mmap mapping(&file, 0);
        const auto * bytes = static_cast<const volatile unsigned char *>(mapping.addr());
        for (size_t i = 0; i < expected.size(); i += page_size) {
            require(bytes[i] == expected[i], "initial mapped data differs");
        }
        require(resident_pages(mapping.addr(), page_size) == std::array<bool, 4>{true, true, true, true},
                "fixture pages were not faulted in");

        // A fragment containing no complete page must not evict adjacent tensor bytes.
        mapping.unmap_fragment(page_size / 2, 3 * page_size / 4);
        mapping.unmap_fragment(page_size, page_size);
        require(resident_pages(mapping.addr(), page_size) == std::array<bool, 4>{true, true, true, true},
                "empty/subpage fragment evicted live boundary data");

        // Pages 0 and 3 contain live boundary bytes; only complete pages 1 and 2 retire.
        mapping.unmap_fragment(page_size / 2, 3 * page_size + page_size / 2);
        require(resident_pages(mapping.addr(), page_size) == std::array<bool, 4>{true, false, false, true},
                "unused mapped pages remained resident or live boundary pages were evicted");
        require(std::memcmp(mapping.addr(), expected.data(), expected.size()) == 0,
                "retiring mapped pages changed file data or broke remapping");

        std::puts("PASS: unused file pages retired; live boundary pages and full readback preserved");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
