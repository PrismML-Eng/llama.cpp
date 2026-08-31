// -----------------------------------------------------------------------------
// Minimal compile-safe patch skeleton for ggml-vulkan.cpp
// -----------------------------------------------------------------------------
// This is a patch-ready extension for the real file:
//   ggml/src/ggml-vulkan/ggml-vulkan.cpp
//
// Purpose:
// - Add a dedicated ternary-matmul dispatch branch
// - Define C++ helper functions and stubs that can be later completed
// - Keep the generic code untouched for non-ternary workloads
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// A. Forward declarations (place near other Vulkan matmul helper declarations)
// -----------------------------------------------------------------------------

static void ggml_vk_mul_mat_ternary(
    ggml_backend_vk_context * ctx,
    vk_context& subctx,
    const struct ggml_cgraph * cgraph,
    int node_idx);

static void ggml_vk_matmul_ternary(
    ggml_backend_vk_context * ctx,
    vk_context& subctx,
    vk_pipeline& pipeline,
    vk_subbuffer&& a,
    vk_subbuffer&& b,
    vk_subbuffer&& d,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t stride_a,
    uint32_t stride_b,
    uint32_t stride_d,
    uint32_t batch_stride_a,
    uint32_t batch_stride_b,
    uint32_t batch_stride_d,
    uint32_t batch,
    uint32_t ne02,
    uint32_t ne12,
    uint32_t broadcast2,
    uint32_t broadcast3,
    uint32_t padded_n);

static vk_pipeline ggml_vk_get_ternary_matmul_pipeline(
    ggml_backend_vk_context * ctx,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    ggml_type src0_type,
    ggml_type src1_type);

// -----------------------------------------------------------------------------
// B. Insert this block at the start of ggml_vk_mul_mat(...)
// -----------------------------------------------------------------------------
// WARNING: this is a code fragment to paste into the function body.
//
// if (src0->type == GGML_TYPE_TERNARY_PACKED || src1->type == GGML_TYPE_TERNARY_PACKED) {
//     ggml_vk_mul_mat_ternary(ctx, subctx, cgraph, node_idx);
//     return;
// }
//
// This must be placed before the generic fallback to ggml_vk_mul_mat_q_f16(...)
// so that the specialized ternary path is selected.

// -----------------------------------------------------------------------------
// C. Minimal implementation of ternary matmul entrypoint
// -----------------------------------------------------------------------------

static void ggml_vk_mul_mat_ternary(
    ggml_backend_vk_context * ctx,
    vk_context& subctx,
    const struct ggml_cgraph * cgraph,
    int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_ternary(" << src0 << ", " << src1 << ", " << dst << ")");

    // Target model is assumed ternary. This is the dedicated path.
    GGML_ASSERT(src0->type == GGML_TYPE_TERNARY_PACKED || src1->type == GGML_TYPE_TERNARY_PACKED);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t ne21 = dst->ne[1];
    const uint32_t stride_d = dst->nb[1] / ggml_type_size(dst->type);
    const uint32_t stride_batch_d = stride_d * ne21;

    const uint64_t r2 = ne12 / ne02;
    const uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *) dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *) src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *) src1->buffer->context;

    vk_buffer d_X = src0_buf_ctx->dev_buffer;
    vk_buffer d_W = src1_buf_ctx->dev_buffer;
    vk_buffer d_D = dst_buf_ctx->dev_buffer;

    const size_t x_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    const size_t w_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    const size_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;

    GGML_ASSERT(d_X != nullptr);
    GGML_ASSERT(d_W != nullptr);
    GGML_ASSERT(d_D != nullptr);

    // This is intentionally a minimal stub: real packed-ternary size is defined by
    // the actual format used by the ternary model loader.
    const uint64_t x_ne = ggml_nelements(src0);
    const uint64_t w_ne = ggml_nelements(src1);
    const uint64_t x_sz = ggml_vk_align_size(sizeof(float) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t w_sz = ggml_vk_align_size(ggml_type_size(src1->type) * w_ne / ggml_blck_size(src1->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);

    vk_pipeline pipeline = ggml_vk_get_ternary_matmul_pipeline(ctx, (uint32_t)ne01, (uint32_t)ne11, (uint32_t)ne10, src0->type, src1->type);
    GGML_ASSERT(pipeline != nullptr);

    uint32_t stride_batch_x = ne00 * ne01;
    uint32_t stride_batch_y = ne10 * ne11;

    if (!ggml_vk_dim01_contiguous(src0)) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }
    if (!ggml_vk_dim01_contiguous(src1)) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    const uint32_t padded_n = ne11;
    const uint32_t split_k = 1;

    ggml_vk_matmul_ternary(
        ctx,
        subctx,
        pipeline,
        { d_X, x_buf_offset, x_sz },
        { d_W, w_buf_offset, w_sz },
        ggml_vk_subbuffer(ctx, d_D, d_buf_offset),
        ne01,
        ne11,
        ne10,
        ne10,
        ne10,
        stride_d,
        stride_batch_x,
        stride_batch_y,
        stride_batch_d,
        split_k,
        ne12 * ne13,
        ne02,
        ne12,
        r2,
        r3,
        padded_n
    );
}

// -----------------------------------------------------------------------------
// D. Minimal ternary matmul dispatcher
// -----------------------------------------------------------------------------

static void ggml_vk_matmul_ternary(
    ggml_backend_vk_context * ctx,
    vk_context& subctx,
    vk_pipeline& pipeline,
    vk_subbuffer&& a,
    vk_subbuffer&& b,
    vk_subbuffer&& d,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t stride_a,
    uint32_t stride_b,
    uint32_t stride_d,
    uint32_t batch_stride_a,
    uint32_t batch_stride_b,
    uint32_t batch_stride_d,
    uint32_t batch,
    uint32_t ne02,
    uint32_t ne12,
    uint32_t broadcast2,
    uint32_t broadcast3,
    uint32_t padded_n) {
    struct vk_ternary_push_constants {
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t stride_a;
        uint32_t stride_b;
        uint32_t stride_d;
        uint32_t batch_stride_a;
        uint32_t batch_stride_b;
        uint32_t batch_stride_d;
        uint32_t batch;
        uint32_t ne02;
        uint32_t ne12;
        uint32_t broadcast2;
        uint32_t broadcast3;
        uint32_t padded_n;
    };

    const vk_ternary_push_constants pc = {
        m, n, k,
        stride_a, stride_b, stride_d,
        batch_stride_a, batch_stride_b, batch_stride_d,
        batch, ne02, ne12, broadcast2, broadcast3, padded_n
    };

    const uint32_t groups_x = CEIL_DIV(m, pipeline->wg_denoms[0]);
    const uint32_t groups_y = CEIL_DIV(n, pipeline->wg_denoms[1]);
    const uint32_t groups_z = batch;

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, pc, { groups_x, groups_y, groups_z });
}

// -----------------------------------------------------------------------------
// E. Minimal ternary pipeline selector
// -----------------------------------------------------------------------------

static vk_pipeline ggml_vk_get_ternary_matmul_pipeline(
    ggml_backend_vk_context * ctx,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    ggml_type src0_type,
    ggml_type src1_type) {
    GGML_UNUSED(m);
    GGML_UNUSED(n);
    GGML_UNUSED(k);
    GGML_UNUSED(src0_type);
    GGML_UNUSED(src1_type);

    // This is the single integrate point for a later AMD-specific ternary shader.
    // The real implementation should return a dedicated device pipeline from
    // ctx->device->pipeline_ternary_matmul_* or similar.
    //
    // Example:
    // return ctx->device->pipeline_ternary_matmul;
    //
    std::cerr << "ggml_vk_get_ternary_matmul_pipeline: ternary pipeline not registered" << std::endl;
    GGML_ABORT("missing ternary Vulkan pipeline");
}

// -----------------------------------------------------------------------------
// F. Drop-in integration patch for ggml_vk_mul_mat(...)
// -----------------------------------------------------------------------------
// Place this code inside ggml_vk_mul_mat(...), before the generic selection logic.
//
//    if (src0->type == GGML_TYPE_TERNARY_PACKED || src1->type == GGML_TYPE_TERNARY_PACKED) {
//        ggml_vk_mul_mat_ternary(ctx, subctx, cgraph, node_idx);
//        return;
//    }
//
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// G. Notes for the actual shader implementation and future follow-up
// -----------------------------------------------------------------------------
// The actual shader should not contain multiplication in the hot loop.
// It should decode packed ternary weights and perform:
//
//    if (code == 1) acc += x;
//    else if (code == -1) acc -= x;
//    else skip;
//
// This is the core reason the ternary backend path is faster for a true ternary
// model on Radeon Pro VII, especially in decode generation.
// -----------------------------------------------------------------------------
