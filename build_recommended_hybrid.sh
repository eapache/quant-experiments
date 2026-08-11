#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "usage: $0 BASE_Q2.gguf DONOR_Q4.gguf [OUTPUT.gguf]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base_model=$1
donor_model=$2
output_model=${3:-"$script_dir/results/models/Qwen3.5-4B-Q2-Q4-gate-up-blocks0-1.gguf"}

python3 "$script_dir/build_hybrid_gguf.py" \
    --base "$base_model" \
    --donor "$donor_model" \
    --layers 0 1 \
    --include '\.(ffn_gate|ffn_up)\.' \
    --output "$output_model"
