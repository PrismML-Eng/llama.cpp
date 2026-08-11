#include "hipblaslt_wcache.cuh"
#include <mutex>
#include <vector>

namespace {
std::vector<ggml_hipblaslt_wcache_invalidator> & registry() {
    static std::vector<ggml_hipblaslt_wcache_invalidator> v;   // function-local: no static-init order issue
    return v;
}
std::mutex & registry_mtx() { static std::mutex m; return m; }
}

void ggml_hipblaslt_wcache_register(ggml_hipblaslt_wcache_invalidator fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> lk(registry_mtx());
    registry().push_back(fn);
}

void ggml_hipblaslt_wcache_invalidate(const void * base, size_t size) {
    if (!base || size == 0) return;
    std::lock_guard<std::mutex> lk(registry_mtx());
    for (auto fn : registry()) fn(base, size);
}
