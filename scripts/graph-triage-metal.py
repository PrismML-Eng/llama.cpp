#!/usr/bin/env python3
"""Decode-graph bottleneck triage for Bonsai-27B Q1_0 on M5 Pro.

Model calibrated ENTIRELY from this session's measurements:
  - DRAM ceiling 251 GB/s (dispatch-size sweep, 179MB dispatch)
  - fixed per-barrier-separated-dispatch cost ~6.8us (the sweep fits
    t = 6.8us + bytes/251GB/s across 1.8..179MB with <2% error)
  - compute ceiling 31 TFLOPS fp16 tensor-units (MLX GEMM bench); decode matvecs are
    BW-bound so this rarely binds
Node time model: t = overhead_share + max(bytes/251, flops/31e12)
  where overhead_share = 6.8us for barrier-group leaders, 0 for concurrent followers.
  We approximate group structure with the measured barrier count (1022 barriers / 1622 nodes
  => ~63% of nodes lead a group; we scale per-node overhead by that fraction).
"""
import sys
from collections import defaultdict

PATH = "/private/tmp/claude-501/-Users-brian/46773ed8-a174-4a73-9c90-996e0120aefb/scratchpad/graph_dump_decode.tsv"

BW = 251e9           # B/s
FLOPS = 31e12        # FLOP/s
OVERHEAD = 6.8e-6    # s per barrier-separated dispatch
BARRIER_FRAC = 1022/1622  # measured barriers / nodes (binary decode graph, h_safe fix on)

TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "i32": 4.0, "i64": 8.0,
    "q1_0": 18.0/128, "q2_0": 34.0/128, "q8_0": 34.0/32, "q4_0": 18.0/32,
}

def nbytes(dims, ty):
    n = 1
    for d in dims: n *= d
    return n * TYPE_BYTES.get(ty, 4.0)

def parse_tensor(spec):
    if spec == "-": return None
    dims_s, ty = spec.rsplit(":", 1)
    return [int(x) for x in dims_s.split(",")], ty

rows = []
for line in open(PATH):
    p = line.rstrip("\n").split("\t")
    if len(p) < 7: continue
    _, idx, op, name, dst_s, s0_s, s1_s = p
    dst = parse_tensor(dst_s); s0 = parse_tensor(s0_s); s1 = parse_tensor(s1_s)
    rows.append((int(idx), op, name, dst, s0, s1))

def classify(op, name):
    n = name.lower()
    if op == "MUL_MAT":
        if "ffn_gate" in n or "ffn_up" in n: return "MLP gate/up matvec"
        if "ffn_out" in n or "ffn_down" in n: return "MLP down matvec"
        if "result_output" in n or n == "output" or "output.weight" in n: return "lm_head matvec"
        return "attn/GDN-input matvec"
    if op == "GATED_DELTA_NET": return "GDN recurrence kernel"
    if op == "FLASH_ATTN_EXT": return "flash attention"
    if op == "SSM_CONV": return "GDN conv"
    if op in ("RMS_NORM", "NORM"): return "norms"
    if op in ("MUL", "ADD", "SUB", "DIV", "SCALE", "GLU", "UNARY", "SIGMOID", "SOFTPLUS", "EXP"): return "elementwise"
    if op in ("CPY", "CONT", "SET_ROWS", "GET_ROWS", "CONCAT"): return "copies/gathers"
    if op == "ROPE": return "rope"
    return f"other ({op})"

agg_t = defaultdict(float); agg_bytes = defaultdict(float); agg_n = defaultdict(int); agg_ovh = defaultdict(float)
total_t = 0.0
biggest = []

for idx, op, name, dst, s0, s1 in rows:
    b = 0.0; fl = 0.0
    if dst: b += nbytes(*dst)
    if s0:  b += nbytes(*s0)
    if s1:  b += nbytes(*s1)
    if op == "MUL_MAT" and s0 and s1:
        (d0, t0), (d1, _) = s0, s1
        fl = 2.0 * d0[0] * d0[1] * d1[1] * max(1, d1[2]) * max(1, d1[3])
    var_t = max(b / BW, fl / FLOPS)
    ovh = OVERHEAD * BARRIER_FRAC
    t = ovh + var_t
    cls = classify(op, name)
    agg_t[cls] += t; agg_bytes[cls] += b; agg_n[cls] += 1; agg_ovh[cls] += ovh
    total_t += t
    biggest.append((t, idx, op, name, b, fl))

print(f"modeled total: {total_t*1000:.2f} ms/token   (measured: ~25.3 ms at 39.5 t/s)")
print(f"  streaming share: {sum(agg_bytes.values())/BW*1000:.2f} ms   overhead share: {sum(agg_ovh.values())*1000:.2f} ms")
print()
print(f"{'class':30s} {'nodes':>6s} {'MB moved':>9s} {'time ms':>8s} {'% of total':>10s}")
for cls in sorted(agg_t, key=lambda c: -agg_t[c]):
    print(f"{cls:30s} {agg_n[cls]:6d} {agg_bytes[cls]/1e6:9.1f} {agg_t[cls]*1000:8.3f} {agg_t[cls]/total_t*100:9.1f}%")

print("\ntop 12 individual nodes:")
for t, idx, op, name, b, fl in sorted(biggest, reverse=True)[:12]:
    print(f"  {t*1e6:8.1f} us  [{idx:4d}] {op:<14s} {name:<28s} {b/1e6:8.2f} MB  {fl/1e9:6.1f} GFLOP")
