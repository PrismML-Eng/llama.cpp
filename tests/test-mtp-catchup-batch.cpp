// Host-only bounds for draft-mtp's first decode: deferred catch-up plus one
// anchor per drafting sequence must fit in llama_n_batch(ctx_dft).
#include "speculative.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

int main() {
    // full-prefill boundary: n_max=31, n_batch=32, one 32-token prompt stash
    assert(common_speculative_mtp_first_decode_fits(32, 31, 1));
    assert(!common_speculative_mtp_first_decode_fits(32, 32, 1));

    // two sequences, combined catch-up plus two anchors
    assert(common_speculative_mtp_first_decode_fits(32, 15 + 15, 2));
    assert(!common_speculative_mtp_first_decode_fits(32, 16 + 16, 2));

    assert(!common_speculative_mtp_first_decode_fits(0, 0, 0));
    assert(!common_speculative_mtp_first_decode_fits(32, -1, 1));

    printf("ok\n");
    return 0;
}
