from pathlib import Path
import re

src = Path('build/ggml/src/ggml-vulkan/mul_mat_vec.comp.cpp')
out = Path('build/ggml/src/ggml-vulkan/mul_mat_vec_q1_0_g128_f32_f32.spv')

if not src.exists():
    raise FileNotFoundError(f'Missing generated shader source: {src}')

text = src.read_text(encoding='utf-8', errors='ignore')
pat = re.compile(
    r'const\s+uint64_t\s+mul_mat_vec_q1_0_g128_f32_f32_len\s*=\s*(\d+)\s*;\s*'
    r'const\s+unsigned\s+char\s+mul_mat_vec_q1_0_g128_f32_f32_data\s*\[[^\]]*\]\s*=\s*\{(.*?)\};',
    re.S,
)
m = pat.search(text)
if not m:
    raise RuntimeError('Could not find the generated SPIR-V blob for mul_mat_vec_q1_0_g128_f32_f32 in mul_mat_vec.comp.cpp')

length = int(m.group(1))
bytes_text = m.group(2)
nums = re.findall(r'0x[0-9A-Fa-f]+', bytes_text)
if len(nums) < length:
    raise RuntimeError(f'Expected at least {length} bytes but found {len(nums)} hex values in shader blob')
raw = bytes(int(x, 16) for x in nums[:length])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(raw)
print(f'WROTE {out}')
print(f'SIZE {len(raw)} bytes')
print(f'HEADER {raw[:16].hex()}')
