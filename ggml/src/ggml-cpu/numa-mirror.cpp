// NUMA weight mirroring for repacked CPU tensors (opt-in: GGML_NUMA_MIRROR=1).
//
// On multi-socket hosts, interleaved weight pages send ~1/2 of every decode
// read across the socket link, which caps throughput at link bandwidth long
// before the combined memory channels saturate. This keeps one copy of each
// repacked weight tensor per NUMA node (mmap + mbind, no libnuma dependency)
// and lets each thread read the copy local to the cpu it is currently running
// on — locality holds even if threads migrate, and the dynamic chunk scheduler
// needs no changes. Copies are made after repack, so results are bit-exact.
#include "numa-mirror.h"

#if defined(__linux__)

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>

#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define GGML_NUMA_MIRROR_MAX_NODES 8

struct ggml_numa_mirror_topo {
    int  n_nodes = 0;
    bool enabled = false;
    std::array<int8_t, 4096> cpu_node{};
};

static ggml_numa_mirror_topo & ggml_numa_mirror_get_topo() {
    static ggml_numa_mirror_topo topo = [] {
        ggml_numa_mirror_topo t;
        t.cpu_node.fill(0);
        const char * env = getenv("GGML_NUMA_MIRROR");
        if (!env || env[0] == '\0' || env[0] == '0') {
            return t;
        }
        for (int n = 0; n < GGML_NUMA_MIRROR_MAX_NODES; ++n) {
            char path[128];
            snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
            FILE * f = fopen(path, "r");
            if (!f) {
                break;
            }
            char buf[512] = { 0 };
            if (fgets(buf, sizeof(buf), f)) {
                char * p = buf;
                while (*p && *p != '\n') {
                    char * end = nullptr;
                    long a = strtol(p, &end, 10);
                    long b = a;
                    p = end;
                    if (*p == '-') {
                        b = strtol(p + 1, &end, 10);
                        p = end;
                    }
                    for (long c = a; c <= b && c < (long) t.cpu_node.size(); ++c) {
                        t.cpu_node[c] = (int8_t) n;
                    }
                    if (*p == ',') {
                        ++p;
                    }
                }
            }
            fclose(f);
            t.n_nodes = n + 1;
        }
        t.enabled = t.n_nodes > 1;
        if (t.enabled) {
            fprintf(stderr, "numa-mirror: enabled, %d nodes\n", t.n_nodes);
        }
        return t;
    }();
    return topo;
}

struct ggml_numa_mirror_entry {
    size_t size = 0;
    void * copy[GGML_NUMA_MIRROR_MAX_NODES] = { nullptr };
};

// Writers (register/free, load-time only) rebuild an immutable snapshot under
// a mutex; the compute-path reader does a plain atomic load + lookup on the
// immutable map - no shared lock, no writer-visible cacheline traffic.
using ggml_numa_mirror_map_t = std::unordered_map<const void *, ggml_numa_mirror_entry>;

static std::mutex & ggml_numa_mirror_mutex() {
    static std::mutex m;
    return m;
}
static ggml_numa_mirror_map_t & ggml_numa_mirror_registry() {
    static ggml_numa_mirror_map_t m; // master copy, guarded by the mutex
    return m;
}
static std::atomic<const ggml_numa_mirror_map_t *> & ggml_numa_mirror_snapshot() {
    static std::atomic<const ggml_numa_mirror_map_t *> s{nullptr};
    return s;
}
// must be called with the mutex held; old snapshots are intentionally leaked
// (bytes, not mirrors - readers may still hold them, load happens rarely)
static void ggml_numa_mirror_publish() {
    ggml_numa_mirror_snapshot().store(new ggml_numa_mirror_map_t(ggml_numa_mirror_registry()),
                                      std::memory_order_release);
}

static bool ggml_numa_mirror_bind(void * p, size_t size, int node) {
    unsigned long mask = 1UL << node;
    // MPOL_BIND = 2; pages are still unfaulted, so they allocate on `node` at
    // the memcpy below
    return syscall(SYS_mbind, p, size, 2, &mask, 8 * sizeof(mask), 0) == 0;
}

bool ggml_numa_mirror_enabled(void) {
    return ggml_numa_mirror_get_topo().enabled;
}

void ggml_numa_mirror_register(const void * base, size_t size) {
    auto & topo = ggml_numa_mirror_get_topo();
    if (!topo.enabled || base == nullptr || size == 0) {
        return;
    }

    ggml_numa_mirror_entry e;
    bool exists = false;
    {
        std::lock_guard lock(ggml_numa_mirror_mutex());
        auto it = ggml_numa_mirror_registry().find(base);
        if (it != ggml_numa_mirror_registry().end()) {
            e      = it->second;
            exists = true;
        }
    }
    if (exists && e.size != size) {
        return; // same address, different tensor size: keep it simple, skip
    }
    if (!exists) {
        e.size = size;
        for (int n = 0; n < topo.n_nodes; ++n) {
            void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED || !ggml_numa_mirror_bind(p, size, n)) {
                if (p != MAP_FAILED) {
                    munmap(p, size);
                }
                for (int m = 0; m < n; ++m) {
                    munmap(e.copy[m], size);
                }
                fprintf(stderr, "numa-mirror: mmap/mbind failed, disabling\n");
                topo.enabled = false;
                return;
            }
            e.copy[n] = p;
        }
    }
    for (int n = 0; n < topo.n_nodes; ++n) {
        memcpy(e.copy[n], base, size);
    }
    {
        std::lock_guard lock(ggml_numa_mirror_mutex());
        ggml_numa_mirror_registry()[base] = e;
        ggml_numa_mirror_publish();
    }
}

const void * ggml_numa_mirror_map(const void * base) {
    auto & topo = ggml_numa_mirror_get_topo();
    if (!topo.enabled) {
        return base;
    }
    const ggml_numa_mirror_map_t * snap = ggml_numa_mirror_snapshot().load(std::memory_order_acquire);
    if (snap == nullptr) {
        return base;
    }
    auto it = snap->find(base);
    if (it == snap->end()) {
        return base;
    }
    int cpu  = sched_getcpu();
    int node = (cpu >= 0 && cpu < (int) topo.cpu_node.size()) ? topo.cpu_node[cpu] : 0;
    return it->second.copy[node];
}

void ggml_numa_mirror_free_range(const void * lo, size_t size) {
    auto & topo = ggml_numa_mirror_get_topo();
    const char * l = (const char *) lo;
    const char * h = l + size;
    std::lock_guard lock(ggml_numa_mirror_mutex());
    for (auto it = ggml_numa_mirror_registry().begin(); it != ggml_numa_mirror_registry().end();) {
        const char * b = (const char *) it->first;
        if (b >= l && b < h) {
            for (int n = 0; n < topo.n_nodes; ++n) {
                if (it->second.copy[n]) {
                    munmap(it->second.copy[n], it->second.size);
                }
            }
            it = ggml_numa_mirror_registry().erase(it);
        } else {
            ++it;
        }
    }
    ggml_numa_mirror_publish();
}

#else // !__linux__

bool ggml_numa_mirror_enabled(void) { return false; }
void ggml_numa_mirror_register(const void * base, size_t size) { (void) base; (void) size; }
const void * ggml_numa_mirror_map(const void * base) { return base; }
void ggml_numa_mirror_free_range(const void * lo, size_t size) { (void) lo; (void) size; }

#endif
