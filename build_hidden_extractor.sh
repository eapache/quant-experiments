#!/usr/bin/env bash
set -euo pipefail

LLAMA_ROOT=${LLAMA_ROOT:-/home/eapache/src/llama.cpp}
CXX=${CXX:-c++}

"$CXX" -std=c++17 -O2 \
  -I"$LLAMA_ROOT/include" \
  -I"$LLAMA_ROOT/src" \
  -I"$LLAMA_ROOT/ggml/include" \
  extract_hidden_states.cpp \
  -L"$LLAMA_ROOT/build/bin" \
  -Wl,-rpath,"$LLAMA_ROOT/build/bin" \
  -lllama -lggml -lggml-base \
  -o extract_hidden_states
