// Verifies vec_dot_ptq1_0_q8_1 numerically on the host by stubbing the CUDA intrinsics.
// nvcc is unavailable here, so this cannot prove the kernel compiles, but it does prove
// the dot-product math and the element mapping agree with the CPU codec on real data.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>

// ---- CUDA intrinsic stubs -------------------------------------------------
static inline int ggml_cuda_dp4a(int a, int b, int c) {          // __dp4a
    const int8_t* pa = (const int8_t*)&a; const int8_t* pb = (const int8_t*)&b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
}
struct half2_stub { float lo, hi; };
static inline float __low2float(half2_stub h) { return h.lo; }

#define QK8_1 32
struct block_q8_1 { half2_stub ds; int8_t qs[QK8_1]; };
static inline int get_int_b4(const int8_t* qs, int j) {           // 4 bytes as an int
    int v; memcpy(&v, qs + j*4, 4); return v;
}

#define QK_PTQ1_0 128
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; float d; };

// ---- transcribed from ggml-cuda/dequantize.cuh ---------------------------
static inline int ptq1_0_trit(const block_ptq1_0 * x, const int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];              n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }
    uint32_t v = b;
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1;
}

// ---- transcribed from ggml-cuda/vecdotq.cuh ------------------------------
static inline float vec_dot_ptq1_0_q8_1(const void* vbq, const block_q8_1* bq8_1,
                                        const int& kbx, const int& iqs) {
    const block_ptq1_0 * bq  = (const block_ptq1_0 *) vbq + kbx;
    const block_q8_1   * bq8 = bq8_1 + iqs;
    const int base = iqs * 32;
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        const int t0 = ptq1_0_trit(bq, base + j*4 + 0);
        const int t1 = ptq1_0_trit(bq, base + j*4 + 1);
        const int t2 = ptq1_0_trit(bq, base + j*4 + 2);
        const int t3 = ptq1_0_trit(bq, base + j*4 + 3);
        const int qx = (t0 & 0xFF) | ((t1 & 0xFF) << 8) | ((t2 & 0xFF) << 16) | ((t3 & 0xFF) << 24);
        const int u  = get_int_b4(bq8->qs, j);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
    }
    return (float) bq->d * __low2float(bq8->ds) * sumi;
}

// ---- reference: dequantize the block, dequantize q8_1, dot in float ------
static void ref_dequant(const block_ptq1_0* x, float* out) {
    const uint8_t pow3[6]={1,3,9,27,81,243}; const size_t st[3]={32,16,8};
    int o=0; size_t j=0;
    for (size_t s=0;s<3;++s){ const size_t c=st[s];
        for(; j+c<=sizeof(x->qs); j+=c)
            for(size_t n=0;n<5;++n) for(size_t m=0;m<c;++m){
                uint8_t q=x->qs[j+m]*pow3[n]; out[o++]=(float)((int)(((uint16_t)q*3)>>8)-1)*x->d; }
    }
    for(size_t n=0;n<4;++n) for(size_t h=0;h<2;++h){
        uint8_t q=x->qh[h]*pow3[n]; out[o++]=(float)((int)(((uint16_t)q*3)>>8)-1)*x->d; }
}

int main(void) {
    unsigned seed=99;
    auto rnd=[&](){ seed=seed*1103515245u+12345u; return (seed>>16)&0xFFFF; };
    double worst_rel = 0.0; long checks = 0; long int_bad = 0, int_checks = 0; double worst_scaled = 0.0;

    for (int trial=0; trial<5000; ++trial) {
        block_ptq1_0 w;
        for (int i=0;i<24;++i) w.qs[i]=rnd()&0xFF;
        for (int i=0;i<2;++i)  w.qh[i]=rnd()&0xFF;
        w.d = 0.01f + (rnd()%1000)/50000.0f;

        block_q8_1 y[4];
        for (int b=0;b<4;++b) {
            y[b].ds.lo = 0.005f + (rnd()%1000)/80000.0f; y[b].ds.hi = 0.f;
            for (int i=0;i<32;++i) y[b].qs[i]=(int8_t)((int)(rnd()%255)-127);
        }

        float wf[QK_PTQ1_0]; ref_dequant(&w, wf);

        // EXACT test: the kernel's integer accumulator per chunk must equal the
        // reference integer sum of trit*q8. This isolates logic from float rounding.
        for (int c = 0; c < 4; ++c) {
            int ref_sumi = 0;
            for (int i = 0; i < 32; ++i) {
                const int trit = (int) llround((double) wf[c*32+i] / (double) w.d);
                ref_sumi += trit * (int) y[c].qs[i];
            }
            // recompute the kernel's sumi by dividing its float result back out
            const float got = vec_dot_ptq1_0_q8_1(&w, y, 0, c);
            const int got_sumi = (int) llround((double) got / ((double) w.d * (double) y[c].ds.lo));
            if (ref_sumi != got_sumi) { ++int_bad; if (int_bad < 4)
                printf("  INT MISMATCH trial %d chunk %d: ref %d got %d\n", trial, c, ref_sumi, got_sumi); }
            ++int_checks;
        }

        // reference dot over all four chunks, in float
        double ref_total = 0.0;
        for (int c=0;c<4;++c)
            for (int i=0;i<32;++i)
                ref_total += (double)wf[c*32+i] * ((double)y[c].qs[i] * (double)y[c].ds.lo);

        // kernel dot, chunk by chunk as MMVQ calls it
        double got_total = 0.0;
        for (int c=0;c<4;++c) got_total += vec_dot_ptq1_0_q8_1(&w, y, 0, c);

        // scale by the sum of magnitudes so cancellation in ref_total cannot inflate it
        double mag = 0.0;
        for (int c=0;c<4;++c) for (int i=0;i<32;++i)
            mag += fabs((double)wf[c*32+i] * (double)y[c].qs[i] * (double)y[c].ds.lo);
        const double denom = fabs(ref_total) > 1e-9 ? fabs(ref_total) : 1.0;
        const double rel = fabs(got_total - ref_total) / denom;
        if (rel > worst_rel) worst_rel = rel;
        if (mag > 0) { const double sc = fabs(got_total - ref_total)/mag; if (sc > worst_scaled) worst_scaled = sc; }
        ++checks;
    }
    printf("  dot products compared : %ld (4 chunks each)\n", checks);
    printf("  worst relative error  : %.3e\n", worst_rel);
    printf("  worst err / sum|terms| : %.3e   (immune to cancellation)\n", worst_scaled);
    printf("  exact integer checks  : %ld, mismatches %ld\n", int_checks, int_bad);
    if (int_bad == 0) printf("  LOGIC EXACT: integer accumulator matches reference on every chunk\n");
    else              printf("  LOGIC BUG in the kernel\n");
    return int_bad != 0;
}
