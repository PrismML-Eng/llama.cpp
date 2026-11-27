## gguf

This is a Python package for writing binary files in the [GGUF](https://github.com/ggml-org/ggml/pull/302)
(GGML Universal File) format.

See [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
as an example for its usage.

## Installation
```sh
pip install gguf
```

Optionally, you can install gguf with the extra 'gui' to enable the visual GGUF editor.
```sh
pip install gguf[gui]
```

## API Examples/Simple Tools

[examples/writer.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/examples/writer.py) — Generates `example.gguf` in the current directory to demonstrate generating a GGUF file. Note that this file cannot be used as a model.

[examples/reader.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/examples/reader.py) — Extracts and displays key-value pairs and tensor details from a GGUF file in a readable format.

[gguf/scripts/gguf_dump.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_dump.py) — Dumps a GGUF file's metadata to the console.

[gguf/scripts/gguf_set_metadata.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_set_metadata.py) — Allows changing simple metadata values in a GGUF file by key.

[gguf/scripts/gguf_convert_endian.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_convert_endian.py) — Allows converting the endianness of GGUF files.

[gguf/scripts/gguf_new_metadata.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_new_metadata.py) — Copies a GGUF file with added/modified/removed metadata values.

[gguf/scripts/gguf_editor_gui.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_editor_gui.py) — Allows for viewing, editing, adding, or removing metadata values within a GGUF file as well as viewing its tensors with a Qt interface.

## Explicit end-of-generation token policy

`GGUFWriter.add_eog_token_ids(ids)` writes the optional
`tokenizer.ggml.eog_token_ids` extension as an array of `UINT32` token IDs.
In the corresponding llama.cpp runtime, presence makes this the complete
end-of-generation set: stop-looking spellings do not add more IDs or change
the vocabulary's declared token types. An explicitly empty array means no
token-based generation stops; omitting the key retains legacy inference.
IDs must be within the model vocabulary. This policy is independent of BOS
insertion and lexical/special-token parsing.

Older runtimes may ignore this extension. Use a runtime that implements it
when canonical stop behavior matters. Empty arrays retain their element
type through the Python writer and reader.

For Maple conversion, the producer validates the exact NFC/BPE tokenizer family and added-token boundary flags, preserves each token's declared lexical/special status, and writes the generation configuration's complete EOS policy. The producer converts the reference sliding window's left span to GGUF's total causal-key count (`left_span + 1`); the runtime must not repeat this adjustment. Model saving retains the current EOG set (including empty) and NFC policy.

Native regression coverage is registered as `test-tokenizer-eog-policy`, `test-tokenizer-eog-roundtrip`, and `test-tokenizer-normalization-{gpt2,qwen35}`. The first two use synthetic fixtures; the normalization variants use the matching vocabulary GGUF under `models/`.

## Development
Maintainers who participate in development of this package are advised to install it in editable mode:

```sh
cd /path/to/llama.cpp/gguf-py

pip install --editable .
```

**Note**: This may require to upgrade your Pip installation, with a message saying that editable installation currently requires `setup.py`.
In this case, upgrade Pip to the latest:

```sh
pip install --upgrade pip
```

## Automatic publishing with CI

There's a GitHub workflow to make a release automatically upon creation of tags in a specified format.

1. Bump the version in `pyproject.toml`.
2. Create a tag named `gguf-vx.x.x` where `x.x.x` is the semantic version number.

```sh
git tag -a gguf-v1.0.0 -m "Version 1.0 release"
```

3. Push the tags.

```sh
git push origin --tags
```

## Manual publishing
If you want to publish the package manually for any reason, you need to have `twine` and `build` installed:

```sh
pip install build twine
```

Then, follow these steps to release a new version:

1. Bump the version in `pyproject.toml`.
2. Build the package:

```sh
python -m build
```

3. Upload the generated distribution archives:

```sh
python -m twine upload dist/*
```

## Run Unit Tests

From root of this repository you can run this command to run all the unit tests

```bash
python -m unittest discover ./gguf-py -v
```

## TODO
- [ ] Include conversion scripts as command line entry points in this package.
