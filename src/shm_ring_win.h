#pragma once

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Single-producer/single-consumer ABI shared with shm_ring.py.
#pragma pack(push, 1)
struct XessSharedRingHeader {
    char magic[4];
    uint32_t version;
    uint32_t slots;
    uint32_t slotSize;
    volatile LONG64 writeSequence;
    volatile LONG64 readSequence;
    volatile LONG closed;
    volatile LONG error;
    uint8_t reserved[24];
};
struct XessSharedSlotHeader {
    uint32_t packetSize;
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(XessSharedRingHeader) == 64, "shared ring header ABI mismatch");
static_assert(sizeof(XessSharedSlotHeader) == 8, "shared slot header ABI mismatch");

class XessSharedRingReader {
public:
    XessSharedRingReader() = default;
    XessSharedRingReader(const XessSharedRingReader&) = delete;
    XessSharedRingReader& operator=(const XessSharedRingReader&) = delete;
    ~XessSharedRingReader() { close(); }

    bool open(const char* name, uint32_t slots, uint32_t slotSize) {
        close();
        if (!name || slots < 2 || !slotSize) return false;
        std::wstring wideName = widen(name);
        mapping_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wideName.c_str());
        if (!mapping_) return report("OpenFileMappingW");
        view_ = static_cast<uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0));
        if (!view_) return report("MapViewOfFile");
        header_ = reinterpret_cast<XessSharedRingHeader*>(view_);
        if (memcmp(header_->magic, "XSRG", 4) || header_->version != 1 ||
            header_->slots != slots || header_->slotSize != slotSize) {
            fprintf(stderr, "[shm] ring header mismatch\n");
            close();
            return false;
        }
        slots_ = slots;
        slotSize_ = slotSize;
        std::wstring dataName = wideName + L"-data";
        std::wstring spaceName = wideName + L"-space";
        dataEvent_ = OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, dataName.c_str());
        spaceEvent_ = OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, spaceName.c_str());
        if (!dataEvent_ || !spaceEvent_) return report("OpenEventW");
        return true;
    }

    bool read(std::vector<uint8_t>& packet, DWORD timeoutMs = 30000) {
        if (!header_) return false;
        const ULONGLONG deadline = GetTickCount64() + timeoutMs;
        for (;;) {
            const LONG64 writeSequence =
                InterlockedCompareExchange64(&header_->writeSequence, 0, 0);
            const LONG64 readSequence =
                InterlockedCompareExchange64(&header_->readSequence, 0, 0);
            if (readSequence < writeSequence) {
                const uint32_t slot = static_cast<uint32_t>(readSequence % slots_);
                uint8_t* base = view_ + sizeof(XessSharedRingHeader) +
                    static_cast<size_t>(slot) * (sizeof(XessSharedSlotHeader) + slotSize_);
                const auto* slotHeader = reinterpret_cast<const XessSharedSlotHeader*>(base);
                const uint32_t size = slotHeader->packetSize;
                if (!size || size > slotSize_) {
                    fprintf(stderr, "[shm] invalid packet size %u\n", size);
                    InterlockedExchange(&header_->error, 1);
                    return false;
                }
                packet.assign(base + sizeof(XessSharedSlotHeader),
                              base + sizeof(XessSharedSlotHeader) + size);
                MemoryBarrier();
                InterlockedExchange64(&header_->readSequence, readSequence + 1);
                if (!SetEvent(spaceEvent_)) return report("SetEvent(space)");
                return true;
            }
            const ULONGLONG now = GetTickCount64();
            if (now >= deadline) {
                fprintf(stderr, "[shm] timed out waiting for a packet\n");
                return false;
            }
            const DWORD remaining = static_cast<DWORD>(std::min<ULONGLONG>(deadline - now, timeoutMs));
            const DWORD result = WaitForSingleObject(dataEvent_, remaining);
            if (result == WAIT_FAILED) return report("WaitForSingleObject(data)");
            if (result == WAIT_TIMEOUT && GetTickCount64() >= deadline) {
                fprintf(stderr, "[shm] timed out waiting for a packet\n");
                return false;
            }
        }
    }

    void close() {
        header_ = nullptr;
        if (view_) { UnmapViewOfFile(view_); view_ = nullptr; }
        if (dataEvent_) { CloseHandle(dataEvent_); dataEvent_ = nullptr; }
        if (spaceEvent_) { CloseHandle(spaceEvent_); spaceEvent_ = nullptr; }
        if (mapping_) { CloseHandle(mapping_); mapping_ = nullptr; }
        slots_ = slotSize_ = 0;
    }

private:
    static std::wstring widen(const char* value) {
        const int count = MultiByteToWideChar(CP_UTF8, 0, value, -1, nullptr, 0);
        if (count <= 1) return {};
        std::wstring result(static_cast<size_t>(count), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, value, -1, result.data(), count);
        result.resize(static_cast<size_t>(count - 1));
        return result;
    }

    bool report(const char* operation) {
        fprintf(stderr, "[shm] %s failed: %lu\n", operation,
                static_cast<unsigned long>(GetLastError()));
        close();
        return false;
    }

    HANDLE mapping_ = nullptr;
    HANDLE dataEvent_ = nullptr;
    HANDLE spaceEvent_ = nullptr;
    uint8_t* view_ = nullptr;
    XessSharedRingHeader* header_ = nullptr;
    uint32_t slots_ = 0;
    uint32_t slotSize_ = 0;
};
