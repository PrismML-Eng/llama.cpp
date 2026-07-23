// ppc_pack_cache.cpp - tensor-keyed cache for MMA-repacked weights.
//
// Repacking weights into MMA layout is deterministic and depends only on
// the (immutable during inference) weight tensor, so it belongs at model
// load.  Landing it inside ggml's repack.cpp buffer machinery is the
// eventual home; this cache delivers the same behavior at the dispatch
// layer today: the first llamafile_sgemm call for a tensor packs the
// whole matrix once (other threads block on a condvar until it is
// published), and every subsequent call across the model's lifetime
// reuses the packed form.
//
// Keying and safety assumptions (stated for review):
//   * Key = (data pointer, m, k, variant id).  Weight tensors in
//     llama.cpp inference are immutable and their storage is not
//     reallocated, so pointer identity is a sound key for this use.
//     Training / repeated model reloads at the same address would need
//     explicit invalidation -- out of scope and documented.
//   * Capacity-bounded (default 2048 MiB, PPC_MMA_PACK_CACHE_MB to
//     override, 0 disables).  On miss-with-full-cache or allocation
//     failure, acquire returns NULL and callers fall back to the
//     per-call packing path -- behavior is never wrong, only slower.
//   * NO eviction: once full (slots or bytes), new tensors are simply
//     not admitted and their callers use the per-call path.  Eviction
//     was rejected deliberately: an inference working set that exceeds
//     the cap cycles through every tensor once per token, and any
//     evicting policy then re-packs evicted tensors every token --
//     strictly worse than the per-call baseline.  Admission-only makes
//     the cache's effect monotonic: cached tensors win, the rest are
//     exactly at baseline.
//   * A cheap content fingerprint (first+last 32 bytes of the tensor)
//     is folded into the key, so a model unload/reload that lands new
//     weights at the same address does not serve stale packs.

#include "kquants_ppc_mma.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#define PPC_PACK_CACHE_SLOTS 128

typedef struct {
    const void * key;
    uint64_t fp;
    int64_t m, k;
    int     variant;
    void  * buf;
    size_t  bytes;
    int     ready;      // 0 = packing in progress, 1 = usable
    int     used;
} slot_t;

static slot_t g_slots[PPC_PACK_CACHE_SLOTS];
static size_t g_total = 0;

static uint64_t fingerprint(const void * p, int64_t m, int64_t k) {
    // first and last 32 bytes of the raw tensor data; the row extent is
    // conservatively the smallest any format uses (m rows x k/8 bytes).
    const uint8_t * b = (const uint8_t *)p;
    const size_t approx = (size_t)m * (size_t)(k/8);
    const size_t tail = approx >= 32 ? approx - 32 : 0;
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 32; i++) { h ^= b[i]; h *= 1099511628211ull; }
    for (int i = 0; i < 32; i++) { h ^= b[tail + i]; h *= 1099511628211ull; }
    return h;
}
static size_t g_cap   = (size_t)2048 * 1024 * 1024;
static int    g_cap_init = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv = PTHREAD_COND_INITIALIZER;

static void cap_init_locked(void) {
    if (g_cap_init) return;
    g_cap_init = 1;
    const char * e = getenv("PPC_MMA_PACK_CACHE_MB");
    if (e) g_cap = (size_t)strtoull(e, NULL, 10) * 1024 * 1024;
}

extern "C" void * ppc_apack_cache_acquire(const void * key, int64_t m, int64_t k,
                                          int variant, size_t bytes, int * fresh) {
    *fresh = 0;
    pthread_mutex_lock(&g_mu);
    cap_init_locked();
    if (g_cap == 0 || bytes > g_cap) { pthread_mutex_unlock(&g_mu); return NULL; }
    const uint64_t fp = fingerprint(key, m, k);
    for (;;) {
        slot_t * hit = NULL;
        slot_t * stale = NULL;
        for (int i = 0; i < PPC_PACK_CACHE_SLOTS; i++)
            if (g_slots[i].used && g_slots[i].key == key && g_slots[i].m == m &&
                g_slots[i].k == k && g_slots[i].variant == variant) {
                if (g_slots[i].fp == fp) { hit = &g_slots[i]; }
                else { stale = &g_slots[i]; }   // same address, new contents
                break;
            }
        if (stale && stale->ready) {            // reload detected: retire it
            free(stale->buf);
            g_total -= stale->bytes;
            memset(stale, 0, sizeof(*stale));
        } else if (stale) {                     // being packed by another thread
            while (!stale->ready) pthread_cond_wait(&g_cv, &g_mu);
            continue;
        }
        if (hit) {
            while (!hit->ready) pthread_cond_wait(&g_cv, &g_mu);
            void * b = hit->buf;
            pthread_mutex_unlock(&g_mu);
            return b;
        }
        // admission only -- no eviction (see header)
        slot_t * dst = NULL;
        for (int i = 0; i < PPC_PACK_CACHE_SLOTS; i++)
            if (!g_slots[i].used) { dst = &g_slots[i]; break; }
        if (!dst || g_total + bytes > g_cap) { pthread_mutex_unlock(&g_mu); return NULL; }
        void * buf = aligned_alloc(64, bytes);
        if (!buf) { pthread_mutex_unlock(&g_mu); return NULL; }
        dst->key = key; dst->fp = fp; dst->m = m; dst->k = k; dst->variant = variant;
        dst->buf = buf; dst->bytes = bytes; dst->ready = 0; dst->used = 1;
        g_total += bytes;
        *fresh = 1;
        pthread_mutex_unlock(&g_mu);
        return buf;
    }
}

// explicit invalidation for embedders that unload models
extern "C" void ppc_apack_cache_clear(void) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < PPC_PACK_CACHE_SLOTS; i++)
        if (g_slots[i].used && g_slots[i].ready) {
            free(g_slots[i].buf);
            g_total -= g_slots[i].bytes;
            memset(&g_slots[i], 0, sizeof(g_slots[i]));
        }
    pthread_mutex_unlock(&g_mu);
}

extern "C" void ppc_apack_cache_publish(const void * key, int64_t m, int64_t k, int variant) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < PPC_PACK_CACHE_SLOTS; i++)
        if (g_slots[i].used && g_slots[i].key == key && g_slots[i].m == m &&
            g_slots[i].k == k && g_slots[i].variant == variant) { g_slots[i].ready = 1; break; }
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
}
