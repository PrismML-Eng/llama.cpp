#pragma once

#include <stddef.h>

// Opt-in (GGML_NUMA_MIRROR=1) per-NUMA-node mirroring of repacked CPU weight
// tensors: each node holds a local copy made after repack; compute threads
// read the copy local to the cpu they are currently running on. Linux-only,
// no-op elsewhere. See numa-mirror.cpp.
bool         ggml_numa_mirror_enabled(void);
void         ggml_numa_mirror_register(const void * base, size_t size);
const void * ggml_numa_mirror_map(const void * base);
void         ggml_numa_mirror_free_range(const void * lo, size_t size);
