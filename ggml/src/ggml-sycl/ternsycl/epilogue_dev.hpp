// Fused epilogues on the fp32 accumulator, numbered like the xetla vLLM plugin
// and OpenVINO integration (same as TernOCL's epilogue.clh):
//   postop 0 none, 1 silu(acc) * other (SwiGLU gate), 2 acc + other (residual),
//          3 acc + bias[n], 4 sigmoid(acc)
//   f32: C (and bias) in fp32 instead of DT (lm_head logits)
// Unlike the OpenCL -D build, postop and f32 are uniform kernel arguments, so
// one kernel binary serves every epilogue; they only branch at the store.
#pragma once

#include "xe2.hpp"

struct Epi {
    const unsigned short *other;  // DT [M, N]
    const void *bias;             // DT or fp32 [N]
    int postop;
    int f32;
};

template <bool BF16> struct epi_dev {
    using D = xe2::dt<BF16>;

    // xetla_sigmoid; the clamp is a multiply (IGC miscompiles the ?: form in OpenCL)
    static float sigmoid(float x) {
        return sycl::native::recip(sycl::native::exp(-x) + 1.0f) * (float)(x > -10.0f);
    }
    static float apply(float v, float e, int p) {
        if (p == 1) return v * sigmoid(v) * e;
        if (p == 2 || p == 3) return v + e;
        if (p == 4) return sigmoid(v);
        return v;
    }
    // epilogue operand of element i = m * N + n
    static float in(const Epi &e, size_t i, int n) {
        if (e.postop == 1 || e.postop == 2) return D::tof(e.other[i]);
        if (e.postop == 3)
            return e.f32 ? ((const float *)e.bias)[n] : D::tof(((const unsigned short *)e.bias)[n]);
        return 0.0f;
    }
    static void store(void *C, const Epi &e, size_t i, int n, float v) {
        v = apply(v, in(e, i, n), e.postop);
        if (e.f32) ((float *)C)[i] = v;
        else ((unsigned short *)C)[i] = D::from(v);
    }

    // epilogue + 2D block store of one 8 x 16 tile at (m0, n0); out of range
    // rows and columns are zero-filled on read and clipped on write.
    // P / F >= 0 fix postop / f32 at compile time (-1: read them from e).
    template <int P = -1, int F = -1>
    static void store8x16(void *C, const Epi &e, int M, int N, int m0, int n0, xe2::float8 v) {
        using namespace xe2;
        const int postop = P >= 0 ? P : e.postop;
        const bool f32 = F >= 0 ? F : e.f32;
        if (postop == 1 || postop == 2) {
            const ushort8 t = rd_16b_8r16(surf(e.other, N * 2, M, N * 2), n0, m0);
            for (int r = 0; r < 8; ++r) v[r] = apply(v[r], D::tof(t[r]), postop);
        } else if (postop == 3) {
            const float b = f32 ? sycl::bit_cast<float>(rd_32b_1r16(surf(e.bias, N * 4, 1, N * 4), n0, 0))
                                : D::tof(rd_16b_1r16(surf(e.bias, N * 2, 1, N * 2), n0, 0));
            for (int r = 0; r < 8; ++r) v[r] += b;
        } else if (postop == 4) {
            for (int r = 0; r < 8; ++r) v[r] = sigmoid(v[r]);
        }
        if (f32) {
            wr_32b_8r16(surf(C, N * 4, M, N * 4), n0, m0, __builtin_bit_cast(uint8, v));
        } else {
            ushort8 o;
            for (int r = 0; r < 8; ++r) o[r] = D::from(v[r]);
            wr_16b_8r16(surf(C, N * 2, M, N * 2), n0, m0, o);
        }
    }
};
