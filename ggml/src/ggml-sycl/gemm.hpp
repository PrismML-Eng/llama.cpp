//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GEMM_HPP
#define GGML_SYCL_GEMM_HPP

#include "ggml-sycl.h"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

class DnnlGemmWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
#ifdef GGML_SYCL_HAS_BF16
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return dt::bf16;
#endif
        else static_assert(0);
    }

    static void gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        const void * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        void * c, dt ct, const queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        dnnl::memory::dims a_dims = {batches_a, m, k };
        dnnl::memory::dims a_strides = {stra2, stra1, stra0};
        const auto a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

        dnnl::memory::dims b_dims = {batches_b, k, n };
        dnnl::memory::dims b_strides = {strb2, strb0, strb1};
        const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

        dnnl::memory::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::memory::dims c_strides = {m*n, 1,  m };
        const auto c_md    = dnnl::memory::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef GGML_SYCL_F16
        primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#endif

        auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        auto scratchpad_md = matmul_pd.scratchpad_desc();
        auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);

        auto matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_args);
    }

    // dst[t * m + r] = sum_k act[t * k + c] * w[r][c] * scale[c / 128][r]; w is s4, row r contiguous along k.
    static void gemm_s4_group(ggml_backend_sycl_context & ctx, int m, int n, int k, const sycl::half * act,
                              const uint8_t * w4, const float * scales, float * dst, const queue_ptr & q) {
        auto stream = ctx.stream_dnnl(q);
        auto eng    = ctx.engine_dnnl(q);

        const auto act_md = dnnl::memory::desc({ n, k }, dt::f16, tag::ab);
        const auto w_md   = dnnl::memory::desc({ k, m }, dt::s4, tag::ba);
        const auto dst_md = dnnl::memory::desc({ n, m }, dt::f32, tag::ab);
        const auto sc_md  = dnnl::memory::desc({ k / 128, m }, dt::f32, tag::ab);  // oneDNN reads scales as dense ab, whatever the tag

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { 128, 1 }, dt::f32);
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

        auto pd   = dnnl::matmul::primitive_desc(eng, act_md, w_md, dst_md, attr);
        auto prim = dnnl::matmul(pd);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, dnnl::memory(act_md, eng, const_cast<sycl::half *>(act)) });
        args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(w_md, eng, const_cast<uint8_t *>(w4)) });
        args.insert({ DNNL_ARG_DST, dnnl::memory(pd.dst_desc(), eng, dst) });
        args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(sc_md, eng, const_cast<float *>(scales)) });
        args.insert({ DNNL_ARG_SCRATCHPAD, ctx.get_scratchpad_mem(pd.scratchpad_desc(), eng, q) });
        prim.execute(stream, args);
    }

    // dst[t * m + r] = sum_k act[t * k + c] * w[r][c], all int8, int32 accumulate, f32 out (no scales)
    static void gemm_s8(ggml_backend_sycl_context & ctx, int m, int n, int k, const int8_t * act, const int8_t * w,
                        float * dst, const queue_ptr & q) {
        auto       stream = ctx.stream_dnnl(q);
        auto       eng    = ctx.engine_dnnl(q);
        const auto act_md = dnnl::memory::desc({ n, k }, dt::s8, tag::ab);
        const auto w_md   = dnnl::memory::desc({ k, m }, dt::s8, tag::ba);
        const auto dst_md = dnnl::memory::desc({ n, m }, dt::f32, tag::ab);
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        auto pd   = dnnl::matmul::primitive_desc(eng, act_md, w_md, dst_md, attr);
        auto prim = dnnl::matmul(pd);
        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, dnnl::memory(act_md, eng, const_cast<int8_t *>(act)) });
        args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(w_md, eng, const_cast<int8_t *>(w)) });
        args.insert({ DNNL_ARG_DST, dnnl::memory(pd.dst_desc(), eng, dst) });
        args.insert({ DNNL_ARG_SCRATCHPAD, ctx.get_scratchpad_mem(pd.scratchpad_desc(), eng, q) });
        prim.execute(stream, args);
    }

    static void row_gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, const void * b, dt bt, void * c, dt ct, const queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#endif

#endif // GGML_SYCL_GEMM_HPP
