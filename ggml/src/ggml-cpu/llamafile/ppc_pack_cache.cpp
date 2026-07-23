// ppc_pack_cache.cpp — tensor-keyed weight-pack cache.
//
// CANONICAL COPY: this file lives in ppc-mma-kernels/src/ and is
// carried into ggml verbatim by patch 0016.  Edit here, regenerate the
// patch; the unit test (xcheck_cache.cpp, `make test`) runs against
// this exact code, including the N > capacity regime — the regime both
// prior cache defects lived in (REVIEW.md defect log: round-robin
// thrash, then 128-slot admission exhaustion).
//
// Policy, stated for review:
//   * Key = (data pointer, m, k, variant) + a 64-byte content
//     fingerprint that detects model reloads at the same address.
//   * Admission-only, no eviction: an admitted pack lives until
//     ppc_apack_cache_clear().  The cache's effect is monotonic.
//   * The slot table GROWS (doubling, starting at 256): slot count is
//     never the binding constraint.  The byte capacity is the only
//     bound (default 2048 MiB, PPC_MMA_PACK_CACHE_MB overrides, 0
//     disables).
//   * A refusal is NEVER silent: the first refusal and every 64th
//     after it print to stderr with running totals, and the totals are
//     queryable via ppc_apack_cache_stats().  A tensor that packs per
//     call is a performance defect (measured 2-8x tg loss on POWER10,
//     VALIDATION-POWER10.md D3); it must be visible.

#if defined(__has_include)
#  if __has_include("kquants_ppc_mma.h")
#    include "kquants_ppc_mma.h"   // prototypes, when compiled inside ggml
#  endif
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

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

static slot_t * g_slots  = NULL;
static int      g_nslots = 0;
static size_t   g_total  = 0;
static size_t   g_cap    = (size_t)2048 * 1024 * 1024;
static int      g_cap_init = 0;
static size_t   g_admitted = 0;
static size_t   g_refused  = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv = PTHREAD_COND_INITIALIZER;

static uint64_t fingerprint(const void * p, int64_t m, int64_t k) {
    const uint8_t * b = (const uint8_t *)p;
    const size_t approx = (size_t)m * (size_t)(k/8);
    const size_t tail = approx >= 32 ? approx - 32 : 0;
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 32; i++) { h ^= b[i]; h *= 1099511628211ull; }
    for (int i = 0; i < 32; i++) { h ^= b[tail + i]; h *= 1099511628211ull; }
    return h;
}

static void cap_init_locked(void) {
    if (g_cap_init) return;
    g_cap_init = 1;
    const char * e = getenv("PPC_MMA_PACK_CACHE_MB");
    if (e) g_cap = (size_t)strtoull(e, NULL, 10) * 1024 * 1024;
}

static int grow_locked(void) {
    const int n = g_nslots ? g_nslots * 2 : 256;
    slot_t * s = (slot_t *)realloc(g_slots, (size_t)n * sizeof(slot_t));
    if (!s) return 0;
    memset(s + g_nslots, 0, (size_t)(n - g_nslots) * sizeof(slot_t));
    g_slots = s;
    g_nslots = n;
    return 1;
}

static void refuse_locked(size_t bytes) {
    g_refused++;
    if (g_refused == 1 || g_refused % 64 == 0) {
        fprintf(stderr,
            "ppc-mma pack cache: capacity refusal #%zu (%zu MiB requested, "
            "%zu/%zu MiB resident, %zu packs admitted) -- this tensor will "
            "re-pack EVERY call; raise PPC_MMA_PACK_CACHE_MB\n",
            g_refused, bytes >> 20, g_total >> 20, g_cap >> 20, g_admitted);
    }
}

extern "C" void * ppc_apack_cache_acquire(const void * key, int64_t m, int64_t k,
                                          int variant, size_t bytes, int * fresh) {
    *fresh = 0;
    pthread_mutex_lock(&g_mu);
    cap_init_locked();
    if (g_cap == 0 || bytes > g_cap) {
        if (g_cap != 0) refuse_locked(bytes);
        pthread_mutex_unlock(&g_mu);
        return NULL;
    }
    const uint64_t fp = fingerprint(key, m, k);
    for (;;) {
        slot_t * hit = NULL;
        slot_t * stale = NULL;
        for (int i = 0; i < g_nslots; i++)
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
        for (int i = 0; i < g_nslots; i++)
            if (!g_slots[i].used) { dst = &g_slots[i]; break; }
        if (!dst) {
            const int before = g_nslots;
            if (!grow_locked()) { refuse_locked(bytes); pthread_mutex_unlock(&g_mu); return NULL; }
            dst = &g_slots[before];   // first slot the growth added
        }
        if (g_total + bytes > g_cap) { refuse_locked(bytes); pthread_mutex_unlock(&g_mu); return NULL; }
        void * buf = aligned_alloc(64, bytes);
        if (!buf) { refuse_locked(bytes); pthread_mutex_unlock(&g_mu); return NULL; }
        dst->key = key; dst->fp = fp; dst->m = m; dst->k = k; dst->variant = variant;
        dst->buf = buf; dst->bytes = bytes; dst->ready = 0; dst->used = 1;
        g_total += bytes;
        g_admitted++;
        *fresh = 1;
        pthread_mutex_unlock(&g_mu);
        return buf;
    }
}

extern "C" void ppc_apack_cache_publish(const void * key, int64_t m, int64_t k, int variant) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nslots; i++)
        if (g_slots[i].used && g_slots[i].key == key && g_slots[i].m == m &&
            g_slots[i].k == k && g_slots[i].variant == variant) { g_slots[i].ready = 1; break; }
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
}

// explicit invalidation for embedders that unload models
extern "C" void ppc_apack_cache_clear(void) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nslots; i++)
        if (g_slots[i].used && g_slots[i].ready) {
            free(g_slots[i].buf);
            g_total -= g_slots[i].bytes;
            memset(&g_slots[i], 0, sizeof(g_slots[i]));
        }
    pthread_mutex_unlock(&g_mu);
}

// observability: totals since process start
extern "C" void ppc_apack_cache_stats(size_t * admitted, size_t * refused,
                                      size_t * resident_bytes) {
    pthread_mutex_lock(&g_mu);
    if (admitted)       *admitted = g_admitted;
    if (refused)        *refused  = g_refused;
    if (resident_bytes) *resident_bytes = g_total;
    pthread_mutex_unlock(&g_mu);
}
