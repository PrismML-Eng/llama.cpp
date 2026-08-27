#!/usr/bin/env python3
"""Rewrite a legacy arch=dspark drafter GGUF into the arch=dflash convention.

Derived from two known-good pairs (dspark-r16b-block7 and
dspark-btl6l1-binary-cont6k-logtau-24k), where the legacy file and its shipped
-v6 re-export differ only in metadata and tensor names. Tensor data is copied
byte for byte; nothing is requantized.

The transform:
  general.architecture   dspark -> dflash
  dspark.<k>             -> dflash.<k>
  dspark.dspark.<k>      -> dflash.<k>            (legacy double prefix)
  dspark.dspark.mask_token_id -> tokenizer.ggml.mask_token_id
  <arch>.target_layers   += 1 on every element    (see note below)
  tokenizer.*            injected from a donor GGUF (the target model)

  dspark.confidence_head.weight  -> conf_proj.weight
  dspark.confidence_head.bias    -> conf_proj.bias
  dspark.fc.weight               -> fc.weight
  dspark.hidden_norm.weight      -> enc.output_norm.weight
  dspark.markov_head_a.weight    -> markov_w1.weight
  dspark.markov_head_b.weight    -> markov_w2.weight
  dspark.log_snr_fc1/fc2.*       -> log_snr_fc1/fc2.*   (79-tensor family only)

target_layers +1: the runtime taps a layer's INPUT
(llama_set_embeddings_layer_inp), so input of layer k+1 is output of layer k,
which is what these drafters were trained against. Both reference pairs show
the same shift, [1,16,31,46,61] -> [2,17,32,47,62].

usage: dspark_to_dflash.py [--drop-shared-tensors] <legacy.gguf> <donor-with-tokenizer.gguf> <out.gguf>

--drop-shared-tensors: omit token_embd.weight and output.weight from the output.
Both are TENSOR_NOT_REQUIRED on the runtime side; a full-vocab draft borrows the
target's embedding and lm head via ctx_other, which shrinks the drafter by the
two largest tensors (11x total with a Q4_0 repack) at unchanged acceptance. Keep
them only for reduced-vocab drafts or for running the draft on devices the
target does not use.
"""
import os
import shutil
import struct
import sys

GGUF_MAGIC = b"GGUF"

# value type ids
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)

_FIXED = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
          F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}

TENSOR_MAP = {
    "dspark.confidence_head.weight": "conf_proj.weight",
    "dspark.confidence_head.bias":   "conf_proj.bias",
    "dspark.fc.weight":              "fc.weight",
    "dspark.hidden_norm.weight":     "enc.output_norm.weight",
    "dspark.markov_head_a.weight":   "markov_w1.weight",
    "dspark.markov_head_b.weight":   "markov_w2.weight",
    "dspark.log_snr_fc1.weight":     "log_snr_fc1.weight",
    "dspark.log_snr_fc1.bias":       "log_snr_fc1.bias",
    "dspark.log_snr_fc2.weight":     "log_snr_fc2.weight",
    "dspark.log_snr_fc2.bias":       "log_snr_fc2.bias",
}


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        assert self.f.read(4) == GGUF_MAGIC, f"{path}: not a GGUF"
        self.version = struct.unpack("<I", self.f.read(4))[0]
        self.n_tensors = struct.unpack("<Q", self.f.read(8))[0]
        self.n_kv = struct.unpack("<Q", self.f.read(8))[0]
        self.kv = []          # list of (key, type, value)
        for _ in range(self.n_kv):
            k = self._str()
            t = struct.unpack("<I", self.f.read(4))[0]
            self.kv.append((k, t, self._val(t)))
        self.tensors = []     # list of (name, dims, type, offset)
        for _ in range(self.n_tensors):
            nm = self._str()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = [struct.unpack("<Q", self.f.read(8))[0] for _ in range(nd)]
            ty = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors.append((nm, dims, ty, off))
        self.header_end = self.f.tell()
        self.alignment = 32
        for k, t, v in self.kv:
            if k == "general.alignment":
                self.alignment = int(v)
        pad = -self.header_end % self.alignment
        self.data_start = self.header_end + pad

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        if t == STR:
            return self._str()
        if t == ARR:
            et = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            return (et, [self._val(et) for _ in range(n)])
        fmt = _FIXED[t]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]


def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_val(t, v):
    if t == STR:
        return enc_str(v)
    if t == ARR:
        et, items = v
        out = struct.pack("<I", et) + struct.pack("<Q", len(items))
        for it in items:
            out += enc_val(et, it)
        return out
    return struct.pack(_FIXED[t], v)


def transform_kv(src_kv, donor_kv):
    out = []
    seen = set()
    mask_token_id = None

    for k, t, v in src_kv:
        if k == "general.architecture":
            out.append((k, STR, "dflash")); seen.add(k); continue
        if k.startswith("dspark.dspark."):
            rest = k[len("dspark.dspark."):]
            if rest == "mask_token_id":
                mask_token_id = (t, v)
                continue
            nk = "dflash." + rest
        elif k.startswith("dspark."):
            nk = "dflash." + k[len("dspark."):]
        elif k.startswith("tokenizer."):
            # legacy drafters carry only a stub tokenizer.ggml.model = "none", which would
            # win over the donor's real vocab and leave the model with no mask token
            continue
        else:
            nk = k
        if nk.endswith(".target_layers") and t == ARR:
            et, items = v
            v = (et, [int(x) + 1 for x in items])
        out.append((nk, t, v)); seen.add(nk)

    # tokenizer comes from the donor; legacy drafters carry no vocab at all
    for k, t, v in donor_kv:
        if k.startswith("tokenizer.") and k not in seen:
            out.append((k, t, v)); seen.add(k)

    if mask_token_id is not None and "tokenizer.ggml.mask_token_id" not in seen:
        t, v = mask_token_id
        out.append(("tokenizer.ggml.mask_token_id", t, v))

    return out


def main():
    args = sys.argv[1:]
    drop_shared = "--drop-shared-tensors" in args
    if drop_shared:
        args.remove("--drop-shared-tensors")
    if len(args) != 3:
        sys.exit(__doc__)
    src_path, donor_path, out_path = args

    src = Reader(src_path)
    donor = Reader(donor_path)

    arch = dict((k, v) for k, t, v in src.kv).get("general.architecture")
    if arch != "dspark":
        sys.exit(f"refusing: {src_path} has general.architecture={arch!r}, expected 'dspark'")

    new_kv = transform_kv(src.kv, donor.kv)

    SHARED_TENSORS = {"token_embd.weight", "output.weight"}

    # per-tensor data sizes from the offset ordering (last one runs to EOF)
    file_size = os.path.getsize(src_path)
    by_off = sorted(src.tensors, key=lambda t: t[3])
    t_size = {}
    for i, (nm, dims, ty, off) in enumerate(by_off):
        end = by_off[i + 1][3] if i + 1 < len(by_off) else file_size - src.data_start
        t_size[nm] = end - off

    new_tensors = []
    dropped = []
    new_off = 0
    for nm, dims, ty, off in src.tensors:
        nn = TENSOR_MAP.get(nm, nm)
        if nm.startswith("dspark.") and nn == nm:
            sys.exit(f"refusing: unmapped legacy tensor {nm!r}; add it to TENSOR_MAP")
        if drop_shared and nn in SHARED_TENSORS:
            dropped.append(nn)
            continue
        # (name, dims, type, new offset, source offset, size)
        new_tensors.append((nn, dims, ty, new_off, off, t_size[nm]))
        new_off = (new_off + t_size[nm] + src.alignment - 1) // src.alignment * src.alignment

    # header
    hdr = GGUF_MAGIC + struct.pack("<I", src.version)
    hdr += struct.pack("<Q", len(new_tensors)) + struct.pack("<Q", len(new_kv))
    for k, t, v in new_kv:
        hdr += enc_str(k) + struct.pack("<I", t) + enc_val(t, v)
    for nm, dims, ty, noff, _ooff, _sz in new_tensors:
        hdr += enc_str(nm) + struct.pack("<I", len(dims))
        for d in dims:
            hdr += struct.pack("<Q", d)
        hdr += struct.pack("<I", ty) + struct.pack("<Q", noff)

    pad = -len(hdr) % src.alignment
    data_bytes = sum(sz for *_, sz in new_tensors)

    with open(out_path, "wb") as o:
        o.write(hdr)
        o.write(b"\x00" * pad)
        if not drop_shared:
            # contiguous verbatim copy of the whole data section
            src.f.seek(src.data_start)
            shutil.copyfileobj(src.f, o, 1024 * 1024 * 8)
        else:
            blob_start = o.tell()
            for nm, dims, ty, noff, ooff, sz in new_tensors:
                o.seek(blob_start + noff)
                src.f.seek(src.data_start + ooff)
                left = sz
                while left > 0:
                    chunk = src.f.read(min(1024 * 1024 * 8, left))
                    o.write(chunk)
                    left -= len(chunk)

    print(f"  {os.path.basename(src_path)}")
    print(f"    -> {os.path.basename(out_path)}")
    note = f", dropped shared: {', '.join(dropped)}" if dropped else ""
    print(f"    kv {len(src.kv)} -> {len(new_kv)}, tensors {len(new_tensors)}, "
          f"data {data_bytes} bytes copied verbatim{note}")


if __name__ == "__main__":
    main()
