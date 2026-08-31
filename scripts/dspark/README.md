# dspark test fixtures

`tests/test-dspark-loop.cpp` (the Phase 2 block-draft-loop gate) takes its
fixtures on the command line:

    test-dspark-loop <tiny.gguf> <ref.json>

The two scripts here generate them. `build_tiny.py` needs only numpy and
safetensors and runs anywhere; `phase2_py_ref.py` needs torch and the upstream
reference, so it runs in a container.

## Upstream reference

The Python reference is `deepseek-ai/DeepSpec`:

    git clone --depth 1 https://github.com/deepseek-ai/DeepSpec.git

| symbol                       | path                                          |
|------------------------------|-----------------------------------------------|
| `Qwen3DSparkModel`           | `deepspec/modeling/dspark/qwen3/modeling.py`  |
| `sample_draft_token_step`    | `deepspec/modeling/dspark/qwen3/modeling.py`  |
| `forward_dspark_draft_block` | `deepspec/eval/dspark/draft_ops.py`           |
| `_propose`                   | `deepspec/eval/dspark/evaluator.py`           |
| `build_markov_head`          | `deepspec/modeling/dspark/markov_head.py`     |

## Building the fixtures

    python scripts/dspark/build_tiny.py $EXPORT
    python convert_hf_to_gguf.py $EXPORT --outfile $FIX/tiny.gguf --outtype f32

    docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -v $DEEPSPEC:/DeepSpec -v $PWD:/work -v $FIX:/fixtures \
      -w /work -e PYTHONPATH=/DeepSpec nvcr.io/nvidia/pytorch:25.12-py3 \
      bash -c "pip install -q transformers && \
        python scripts/dspark/phase2_py_ref.py /fixtures/export /fixtures/ref.json"

    ./build/bin/test-dspark-loop $FIX/tiny.gguf $FIX/ref.json

`transformers` is not in the NGC image, hence the install. The weights are drawn
from a fixed seed, so the export is reproducible; their values do not matter,
since the gate compares C++ against Python on the same checkpoint.

## Why the driver does not call forward_dspark_draft_block

Upstream slices `position_ids` at absolute positions and calls
`past_key_values.crop(start)`. `common/speculative.cpp` instead keeps drafter
positions inside `[0, window)` and rebases them, so `phase2_py_ref.py` applies
the same window/rebase rule and calls `model._forward_backbone` directly.
Everything else (`embed_tokens`, the backbone, `compute_logits`,
`sample_draft_token_step`) is the reference's own code.

The attended context is exactly `[base, start)`. A rebase drops everything below
`base` from the drafter cache, so restaging from 0 would give those rows negative
rebased positions.

## RoPE positions

dspark does not use the stock helper: it imports `rotate_half` and defines its
own `apply_rotary_pos_emb` (modeling.py):

    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)

`k` takes the full cos/sin; `q` takes the last `q_len` entries. So block queries
get positions `start .. start+block_size-1` and context keys keep
`past_len .. start-1`. This matches the C++ batch, which adds context rows at
`pos[stage0+i] - base` and block rows at `start - base + k`.

## Keeping the gate non-vacuous

`window = min(drafter n_ctx_train, n_batch)`, where `n_ctx_train` is the GGUF
`context_length` (the export's `max_position_embeddings`), and
`test-dspark-loop.cpp` runs at `n_batch = 2*block_size + 9`.

With `block_size = 7`, `n_batch = 23` and the constructor requires
`window > 14`. `max_position_embeddings = 20` gives `window = 20`, `w_keep = 10`,
and against the test's `PROMPT` and `ACCEPT_SCHEDULE`:

    round 0: start=5   5 + 7 = 12 <= 20   no rebase
    round 1: start=13  13 + 7 = 20 <= 20  no rebase
    round 2: start=17  17 + 7 = 24 >  20  rebase

Three rebases fire over one `ACCEPT_SCHEDULE` cycle. If the rounds never cross
the window the gate covers none of the windowed staging logic, so
`test-dspark-loop` counts the rounds `ref.json` marks as rebasing and fails if
none did. Re-check this arithmetic when changing the fixture dims.

## ref.json

Top level: `n_embd_cap`, `block_size`, `vocab_size`, `mask_token_id`,
`prefill_bonus`, `window`, `w_keep`, `rounds`. Each round: `n_accepted`,
`ctx_len`, `start`, `sampled`, `bonus`, `rebase`.

`hash_u32` / `synth_feat` / `synth_bonus_token` are duplicated between
`phase2_py_ref.py` and `test-dspark-loop.cpp` rather than shared: they are
closed-form functions of small integers, not a stateful RNG stream, so bit
parity across languages falls out of using the same uint32 wraparound
arithmetic. `ACCEPT_SCHEDULE` and `PROMPT` are likewise mirrored. Keep both
sides in sync.

## Not covered

`tests/test-dspark-forward.cpp` Tier 2 needs a `ref.bin` from a separate
generator that does not exist in the tree; that gate cannot run.
