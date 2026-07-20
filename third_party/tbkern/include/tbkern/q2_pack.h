/* q2_pack.h - minimal Q2_0 repack contract used by Prism's CPU backend.
 *
 * This is intentionally limited to the cache packer consumed by
 * ggml/src/ggml-cpu/repack.cpp. It is not a general TBKERN kernel API.
 */
#ifndef TBKERN_Q2_PACK_H
#define TBKERN_Q2_PACK_H
#include <stddef.h>
#include <stdint.h>
#include "tbkern/format.h"
#ifdef __cplusplus
extern "C" {
#endif
#define TBK_OK 0
#define TBK_EINVAL -1
#define TBK_ENOMEM -2
#define TBK_ENOTTERNARY -4
typedef enum { TBK_LAYOUT_BITPLANE = 0, TBK_LAYOUT_CODES = 1 } tbk_layout_kind;
typedef struct {
    int32_t M, K, Kp, G;
    tbk_layout_kind layout;
    void * data;
    size_t row_stride_bytes;
    float * scales;
    float * bias;
    int32_t ternary_only;
    void * _owner;
} tbk_mat;
int tbk_pack_from_q2(const uint8_t * gguf_q2_blocks, int M, int K, int G,
                     tbk_layout_kind layout, tbk_mat * out);
void tbk_mat_free(tbk_mat * mat);
#ifdef __cplusplus
}
#endif
#endif
