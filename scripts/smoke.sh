#!/bin/bash
# Smoke test for a running NVFP4 DeepSeek-V4.1-Flash server.
#   ./smoke.sh [port]      (default 8889)
set -euo pipefail
PORT=${1:-8889}
BASE="http://127.0.0.1:${PORT}"
MODEL="deepseek-v4.1-flash-nvfp4"

echo "== /v1/models"
curl -fsS "${BASE}/v1/models" | python3 -m json.tool | head -20

echo "== chat (thinking off, 64 tokens)"
curl -fsS "${BASE}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: NVFP4 Spark OK\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content","").strip())'
