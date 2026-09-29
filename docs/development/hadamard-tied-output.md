# Shared Hadamard embedding and output weights

A latent token embedding can serve both row lookup and the output projection. For a normalized Hadamard matrix `H` and sign vector `s`, lookup reconstructs `s * (H z)`. The output projection uses the same stored rows after transforming its input as `H (s * h)`.

The shared-output contract requires:

- `prism.hadamard.version = 2`
- `prism.hadamard.tied_output = true`
- `token_embd.weight` in `prism.hadamard.inverse_weight_names`
- No `output.weight` tensor or entry in `prism.hadamard.weight_names`

The loader registers the output tensor's activation transform from the embedding's block size and signs. It binds the forward and inverse transforms to the actual output and lookup tensors, which can be separate runtime allocations when placed on different devices. Sharing on disk does not guarantee a single allocation across devices.

Version 1 remains supported for explicit heads. A latent embedding with no explicit head requires version 2; a version-1 runtime rejects version 2 before inference. The graph verifier also rejects a latent lookup tensor consumed by a matrix multiplication without a registered forward transform.

For conversion, a shared latent head requires `hadamard_packing.json` schema 3 with `tied_output=true`, a tied HF configuration, and no separate output head. The embedding retains its `inverse-after-lookup` record. Other folded tensors retain their existing records. Manifest schema versions and GGUF contract versions are distinct; schema 3 makes older converters reject the new contract instead of ignoring the flag.

Plain, non-Hadamard tied embeddings do not need this metadata. Separately trained output heads must remain explicit. Consumers that only support the version-1 Hadamard contract must be updated before accepting version-2 artifacts.
