#!/bin/bash
# =============================================================================
# query.sh -- send a SQL generation request to the running vLLM server.
#
# Usage:
#   bash scripts/query.sh
#   bash scripts/query.sh "<schema>" "<question>"
#   bash scripts/query.sh "<schema>" "<question>" <host> <port> <served-model-name>
#
# All five arguments are optional and positional. The 5th matters when you are
# not serving the default merged checkpoint: serve.sbatch derives
# --served-model-name from `basename "$MODEL_PATH"`, so serving
# output/qwen3.8-27b-sql-full means passing "qwen3.8-27b-sql-full" here, and
# serving the base model means "Qwen3.8-27B". Getting it wrong returns a 404
# from the server, not a useful message.
# =============================================================================
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-/mnt/data/qwen38-demo}"
VENV_PY="${VENV_PY:-$DEMO_DIR/venv/bin/python}"

SCHEMA=${1:-"CREATE TABLE employees (id INT, department TEXT, salary DECIMAL)"}
QUESTION=${2:-"What is the average salary per department?"}
HOST=${3:-$(squeue --noheader -n qwen38-serve -o "%N" 2>/dev/null | head -1)}
PORT=${4:-8000}
MODEL=${5:-"qwen3.8-27b-sql"}   # = basename of serve.sbatch's MODEL_PATH

if [ -z "$HOST" ]; then
    echo "No qwen38-serve job found. Start one with:" >&2
    echo "  sbatch $DEMO_DIR/scripts/serve.sbatch" >&2
    exit 1
fi

# squeue reports an empty NODELIST while the job is PENDING, and vLLM needs
# several minutes to load 51GB and compile, so a bare "sbatch then query" races
# the server. Wait for the endpoint rather than failing with a misleading
# "no job found".
if ! curl -sf -m 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Waiting for http://${HOST}:${PORT} to answer (vLLM load + compile takes ~5 min)..." >&2
    for _ in $(seq 1 60); do
        sleep 10
        curl -sf -m 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 && break
    done
    curl -sf -m 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 || {
        echo "Server at ${HOST}:${PORT} never answered. Check:" >&2
        echo "  tail -f $DEMO_DIR/logs/serve_<JOBID>.err" >&2
        exit 1
    }
fi

echo "Server  : $HOST:$PORT"
echo "Model   : $MODEL"
echo "Schema  : $SCHEMA"
echo "Question: $QUESTION"
echo ""

# Build the request body with json.dumps rather than interpolating into a
# heredoc. Interpolating breaks on any schema containing a double quote or
# backslash -- and `DEFAULT "x"` is ordinary SQL. Measured before this fix:
# the server returned 400 json_invalid "Expecting ',' delimiter".
#
# The system prompt is imported from sft_common rather than copied, so serving
# uses the same prompt as training and evaluation. That import needs the venv
# interpreter, because sft_common imports transformers at module scope.
BODY=$("$VENV_PY" - "$MODEL" "$SCHEMA" "$QUESTION" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ.get("DEMO_DIR", "/mnt/data/qwen38-demo") + "/scripts")
from sft_common import SYSTEM_PROMPT

model, schema, question = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({
    "model": model,
    "temperature": 0,
    "max_tokens": 256,
    # Passed per-request as well as server-side, so this still behaves if the
    # server was started without --default-chat-template-kwargs.
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ],
}))
PYEOF
)

# --fail so an HTTP error is an error, and -m so a hung server does not hang
# the client forever.
RESPONSE=$(curl -sS --fail-with-body -m 120 \
    "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary "$BODY") || {
    echo "Request failed. Server said:" >&2
    echo "$RESPONSE" >&2
    exit 1
}

echo "$RESPONSE" | python3 -c "
import sys, json
raw = sys.stdin.read()
try:
    msg = json.loads(raw)['choices'][0]['message']
except Exception as e:
    print(f'Could not parse server response: {e}', file=sys.stderr)
    print(raw[:2000], file=sys.stderr)
    sys.exit(1)

# With --reasoning-parser qwen3 any thinking lands in reasoning_content, so
# content is the bare SQL. Fall back to stripping tags if the parser was off.
if msg.get('reasoning_content'):
    print('[reasoning suppressed:', len(msg['reasoning_content']), 'chars]', file=sys.stderr)
sql = (msg.get('content') or '').strip()
if '</think>' in sql:
    sql = sql.split('</think>')[-1].strip()
print('SQL:', sql)
"
