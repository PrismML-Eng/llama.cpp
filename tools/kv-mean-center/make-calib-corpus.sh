#!/usr/bin/env bash
# Generate a self-contained calibration corpus for llama-kv-mean-center using the
# model's own chat-templated output, so no external calibration file is needed.
#
# Usage:
#   make-calib-corpus.sh -s <llama-server binary> -m <model.gguf> -o <corpus.txt>
#                        [-n <tokens per prompt, default 600>] [-p <port, default 8901>]
#                        [-g <n-gpu-layers, passed through to llama-server>]
#
# Requires: curl, jq, gzip. Starts a temporary llama-server on the given port,
# generates one response per built-in seed prompt with thinking disabled,
# concatenates the responses, and checks the result is not degenerate (gzip
# compression ratio).
set -euo pipefail

# Print the header comment block (everything between the shebang and the first
# non-comment line) as the help text.
usage() { awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; }

NTOK=600
PORT=8901
SERVER_BIN=""
MODEL=""
OUT=""
NGL=""

while getopts "s:m:o:n:p:g:h" opt; do
  case $opt in
    s) SERVER_BIN=$OPTARG ;;
    m) MODEL=$OPTARG ;;
    o) OUT=$OPTARG ;;
    n) NTOK=$OPTARG ;;
    p) PORT=$OPTARG ;;
    g) NGL=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
  esac
done

if [ -z "$SERVER_BIN" ] || [ -z "$MODEL" ] || [ -z "$OUT" ]; then
  echo "error: -s, -m and -o are required (see -h)" >&2
  exit 1
fi
command -v curl >/dev/null || { echo "error: curl not found" >&2; exit 1; }
command -v jq   >/dev/null || { echo "error: jq not found" >&2; exit 1; }
command -v gzip >/dev/null || { echo "error: gzip not found" >&2; exit 1; }

# Diverse seed prompts: expository, narrative, technical, code, dialogue, instructional.
PROMPTS=(
  "Explain how a hash map works internally, including collision handling."
  "Write a short story about a lighthouse keeper who finds a message in a bottle."
  "Describe the water cycle for a middle-school science class."
  "Write a Python function that merges two sorted lists, with comments."
  "Summarize the causes and consequences of the Industrial Revolution."
  "Explain the difference between TCP and UDP and when to use each."
  "Write a dialogue between a customer and a barista ordering an unusual drink."
  "Describe how photosynthesis converts light energy into chemical energy."
  "Give step-by-step instructions for making fresh pasta from scratch."
  "Explain what a stock index is and how index funds work."
  "Write a product review for a fictional pair of noise-cancelling headphones."
  "Explain recursion with two concrete examples, one numeric and one on trees."
  "Describe the rules of chess to someone who has never played."
  "Write a persuasive paragraph arguing for more urban green spaces."
  "Explain how vaccines train the immune system."
  "Write a SQL tutorial snippet covering JOINs with a small example schema."
  "Describe a day in the life of a marine biologist studying coral reefs."
  "Explain the doppler effect and give everyday examples."
  "Write a cover letter for a junior software engineering position."
  "Explain how compilers optimize loops, mentioning unrolling and vectorization."
  "Describe the history and cultural significance of tea in East Asia."
  "Write a troubleshooting guide for a home Wi-Fi connection that keeps dropping."
  "Explain probability with coin flips and dice, including expected value."
  "Write a scene where two old friends meet by chance at a train station."
)

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "starting llama-server on port $PORT ..." >&2
# NGL is optional; when unset the server's own --n-gpu-layers default applies.
"$SERVER_BIN" -m "$MODEL" ${NGL:+-ngl "$NGL"} --port "$PORT" >/dev/null 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "error: server exited during startup" >&2; exit 1; fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "error: server not ready after 240s" >&2; exit 1; }

: > "$OUT"
i=0
for p in "${PROMPTS[@]}"; do
  i=$((i+1))
  echo "[$i/${#PROMPTS[@]}] $p" >&2
  # chat_template_kwargs disables thinking where the template supports it; harmless otherwise.
  jq -n --arg p "$p" --argjson n "$NTOK" \
    '{messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0.9,top_p:0.95,chat_template_kwargs:{enable_thinking:false}}' \
  | curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- \
  | jq -r '.choices[0].message | ((.content // "") + (if (.reasoning_content // "") != "" and (.content // "") == "" then .reasoning_content else "" end))' >> "$OUT"
  printf '\n\n' >> "$OUT"
done

RAW=$(wc -c < "$OUT")
GZ=$(gzip -c "$OUT" | wc -c)
RATIO=$(awk -v r="$RAW" -v g="$GZ" 'BEGIN { printf "%.2f", r / g }')
echo "corpus: $RAW bytes, gzip ratio $RATIO" >&2

if [ "$RAW" -lt 20000 ]; then
  echo "error: corpus too small ($RAW bytes); is the model generating?" >&2
  exit 1
fi
# Natural prose lands around 2.2-3.0; much higher means repetitive/degenerate output,
# which under-excites K channels and gives a poor mean estimate.
if awk -v x="$RATIO" 'BEGIN { exit !(x > 4.0) }'; then
  echo "error: corpus looks degenerate (gzip ratio $RATIO > 4.0); raise temperature or check the model" >&2
  exit 1
fi

echo "wrote $OUT" >&2
