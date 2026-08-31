// -----------------------------------------------------------------------------
// Ternary Vulkan patch skeleton for direct add/sub matmul path
// -----------------------------------------------------------------------------
// This file is intentionally a patch skeleton for insertion into
// ggml/src/ggml-vulkan/ggml-vulkan.cpp.
//
// Assumption: target model is ternary by construction.
// We bypass the generic quantized matmul path and dispatch an explicit
// ternary matmul pipeline that works by add/sub instead of multiply.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 1) New helper declarations near other Vulkan matmul helpers
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
// 2) Exact dispatch hook inside ggml_vk_mul_mat(...)
// -----------------------------------------------------------------------------
// Insert before the generic fallback block, ideally right after the needs_split
// logic and before the fallback to ggml_vk_mul_mat_q_f16(...)
//
// Example insertion:
//
//     if (src0->type == GGML_TYPE_TERNARY_PACKED || src1->type == GGML_TYPE_TERNARY_PACKED) {
//         ggml_vk_mul_mat_ternary(ctx, subctx, cgraph, node_idx);
//         return;
//     }
//
// This must be placed before the generic vector/matmul selection flow.

// -----------------------------------------------------------------------------
// 3) Ternary matmul entrypoint
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

    // AXIOM: target workload is ternary.
    // This path is intentionally specialized and does not use the generic
    // dequant->fp16->mul_mat pipeline.

    // -------------------------------------------------------------------------
    // Validate the assumed ternary layout.
    // Replace the assumptions with the actual format used by the model loader.
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Input buffers.
    // -------------------------------------------------------------------------
    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *) dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *) src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *) src1->buffer->context;

    vk_buffer d_X = nullptr;
    vk_buffer d_W = nullptr;
    vk_buffer d_D = nullptr;

    size_t x_buf_offset = 0;
    size_t w_buf_offset = 0;
    size_t d_buf_offset = 0;

    // -------------------------------------------------------------------------
    // Map storage.
    // -------------------------------------------------------------------------
    d_X = src0_buf_ctx->dev_buffer;
    d_W = src1_buf_ctx->dev_buffer;
    d_D = dst_buf_ctx->dev_buffer;

    x_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    w_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;

    GGML_ASSERT(d_X != nullptr);
    GGML_ASSERT(d_W != nullptr);
    GGML_ASSERT(d_D != nullptr);

    // -------------------------------------------------------------------------
    // This is the rough size estimate for the packed ternary format.
    // Replace with actual packed-size computation once format is defined.
    // -------------------------------------------------------------------------
    const uint64_t x_ne = ggml_nelements(src0);
    const uint64_t w_ne = ggml_nelements(src1);
    const uint64_t d_ne = ggml_nelements(dst);

    const uint64_t x_sz = ggml_vk_align_size(sizeof(float) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t w_sz = ggml_vk_align_size(ggml_type_size(src1->type) * w_ne / ggml_blck_size(src1->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t d_sz = sizeof(float) * d_ne;

    // -------------------------------------------------------------------------
    // Request descriptor sets for ternary shader.
    // -------------------------------------------------------------------------
    vk_pipeline pipeline = ggml_vk_get_ternary_matmul_pipeline(ctx, (uint32_t)ne01, (uint32_t)ne11, (uint32_t)ne10, src0->type, src1->type);
    GGML_ASSERT(pipeline != nullptr);

    // -------------------------------------------------------------------------
    // The ternary matmul is a direct path; do not convert the weights to fp16.
    // -------------------------------------------------------------------------
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
// 4) Ternary matmul dispatch helper
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
    // -------------------------------------------------------------------------
    // This is the direct ternary dispatch replacement for generic matmul.
    // The actual shader will read packed ternary weights and accumulate using
    // add/sub instead of multiply.
    // -------------------------------------------------------------------------
    VK_LOG_DEBUG("ggml_vk_matmul_ternary(...)");

    // The push constants must match the shader layout exactly.
    // We use a minimal set here; extend as needed.
    struct vk_ternary_matmul_push_constants {
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

    const vk_ternary_matmul_push_constants pc = {
        m, n, k,
        stride_a, stride_b, stride_d,
        batch_stride_a, batch_stride_b, batch_stride_d,
        batch, ne02, ne12, broadcast2, broadcast3, padded_n
    };

    // Dispatch pattern intentionally simple; it can be refined per AMD tuning.
    const uint32_t groups_x = CEIL_DIV(m, pipeline->wg_denoms[0]);
    const uint32_t groups_y = CEIL_DIV(n, pipeline->wg_denoms[1]);
    const uint32_t groups_z = batch;

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, pc, { groups_x, groups_y, groups_z });
}

// -----------------------------------------------------------------------------
// 5) Ternary pipeline selector
// -----------------------------------------------------------------------------

static vk_pipeline ggml_vk_get_ternary_matmul_pipeline(
    ggml_backend_vk_context * ctx,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    ggml_type src0_type,
    ggml_type src1_type) {
    GGML_UNUSED(src0_type);
    GGML_UNUSED(src1_type);

    // NOTE:
    // This is intentionally a stub. The real implementation should select among
    // a dedicated ternary matmul pipeline family, likely with AMD-specific tuning.
    //
    // For example the selector may prefer:
    //  - small decode M/N tiles for short decode
    //  - larger tiles for long K and wide outputs
    //  - separate shader variants for M=1 / M small / general case
    //
    // For the initial patch, it should return the correct existing pipeline if a
    // specialized shader is already registered, else trigger an explicit abort.

    // Example placeholder:
    // return ctx->device->pipeline_ternary_matmul;

    std::cerr << "ggml_vk_get_ternary_matmul_pipeline: ternary pipeline not implemented yet" << std::endl;
    GGML_ABORT("ternary Vulkan matmul pipeline missing");
}

// -----------------------------------------------------------------------------
// 6) Example of a new shader entry in the Vulkan shader collection
// -----------------------------------------------------------------------------
// File suggestion:
//   ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_ternary.comp
//
// The shader should execute the following conceptual operation:
//
//   for each output element (row, col):
//      acc = 0
//      for k in K_block:
//          code = decode_ternary_weight(weight_block[k])
//          x = load_input(row, k)
//          if code == +1: acc += x;
//          else if code == -1: acc -= x;
//          else: skip;
//      output = acc
//
// This is the direct replacement for multiply-based matmul.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 7) Example of a minimal AMD-specific tuning hook
// -----------------------------------------------------------------------------
// This placeholder should be used as a guide when adding device-specific config
// for Radeon Pro VII.
//
// Example logic:
//
// if (ctx->device->architecture == AMD_GCN) {
//     // Prefer wider output tiles for decode and small M
//     // Use smaller workgroup sizes for short decode
//     // Reduce split-K in decode path
// }
//
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 8) Example fused decode sequence (optional later step)
// -----------------------------------------------------------------------------
// After the direct ternary matmul works, the next optimization is to fuse:
//   RMSNorm -> Scale -> TernaryMatMul -> ResidualAdd
// into fewer Vulkan dispatches.
//
// This reduces overhead and helps decode generation on Radeon Pro VII.
// -----------------------------------------------------------------------------
