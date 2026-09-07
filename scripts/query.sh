#!/bin/bash
# =============================================================================
# query.sh -- send a SQL generation request to the running vLLM server.
#
# Usage:
#   bash scripts/query.sh
#   bash scripts/query.sh "CREATE TABLE orders (id INT, total DECIMAL)" \
#                         "What is the sum of all totals?"
#   bash scripts/query.sh "<schema>" "<question>" worker-b300-0 8001
# =============================================================================
set -uo pipefail

SCHEMA=${1:-"CREATE TABLE employees (id INT, department TEXT, salary DECIMAL)"}
QUESTION=${2:-"What is the average salary per department?"}
HOST=${3:-$(squeue --noheader -n qwen38-serve -o "%N" 2>/dev/null | head -1)}
PORT=${4:-8000}
MODEL=${5:-"qwen3.8-27b-sql"}   # matches --served-model-name in serve.sbatch

if [ -z "$HOST" ]; then
    echo "No qwen38-serve job found. Start one with:"
    echo "  sbatch /mnt/data/qwen38-demo/scripts/serve.sbatch"
    exit 1
fi

echo "Server  : $HOST:$PORT"
echo "Model   : $MODEL"
echo "Schema  : $SCHEMA"
echo "Question: $QUESTION"
echo ""

# enable_thinking:false is passed per-request as well as server-side, so this
# still behaves if the server was started without the default.
RESPONSE=$(curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @- <<EOF
{
  "model": "$MODEL",
  "temperature": 0,
  "max_tokens": 256,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [
    {"role": "system", "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query. Output only the SQL query, nothing else."},
    {"role": "user", "content": "Schema:\n$SCHEMA\n\nQuestion: $QUESTION"}
  ]
}
EOF
)

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
