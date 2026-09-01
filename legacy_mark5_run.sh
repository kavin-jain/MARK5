#!/usr/bin/env bash
# ╔════════════════════════════════════════════════════════════════╗
# ║          MARK5 AUTONOMOUS ORCHESTRATOR v1.0                  ║
# ║          OODA / L99 / GODMODE — Runs until goal met          ║
# ╚══════════════════════════════════════════════════════════════╝
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Args ─────────────────────────────────────────────────────
GOAL="${1:-}"
MAX_ITER="${2:-999}"
DELAY="${3:-8}"  # seconds between iterations

if [ -z "$GOAL" ]; then
  echo -e "${RED}Usage: ./mark5_run.sh \"your goal here\" [max_iterations] [delay_seconds]${RESET}"
  echo -e "${YELLOW}Example: ./mark5_run.sh \"Write a Python web scraper\" 50 5${RESET}"
  exit 1
fi

# ── Directories ───────────────────────────────────────────────
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$HOME/.mark5/runs/$RUN_ID"
STATE_FILE="$LOG_DIR/state.json"
MEMORY_FILE="$LOG_DIR/memory.md"
RESULT_FILE="$LOG_DIR/final_result.md"
mkdir -p "$LOG_DIR"

# ── Banner ────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   MARK5 AUTONOMOUS ORCHESTRATOR — OODA/L99/GODMODE  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "${BOLD}Goal:${RESET} $GOAL"
echo -e "${BOLD}Max Iterations:${RESET} $MAX_ITER"
echo -e "${BOLD}Delay:${RESET} ${DELAY}s"
echo -e "${BOLD}Log:${RESET} $LOG_DIR"
echo ""

# ── Init State ────────────────────────────────────────────────
python3 - <<PYEOF
import json
state = {
    "goal": "$GOAL",
    "run_id": "$RUN_ID",
    "iteration": 0,
    "confidence": 0,
    "done": False,
    "history": [],
    "best_attempt": "",
    "stuck_count": 0
}
with open("$STATE_FILE", "w") as f:
    json.dump(state, f, indent=2)
PYEOF

cat > "$MEMORY_FILE" <<EOF
# MARK5 Memory — Run $RUN_ID
**Goal:** $GOAL
**Started:** $(date)

---
EOF

# ── Helper: Call Gemini ───────────────────────────────────────
call_gemini() {
  local prompt="$1"
  # Try gemini CLI — supports both 'gemini' and 'gemini-cli' commands
  if command -v gemini &>/dev/null; then
    echo "$prompt" | gemini 2>/dev/null
  elif command -v gemini-cli &>/dev/null; then
    echo "$prompt" | gemini-cli 2>/dev/null
  else
    echo '{"error":"gemini_not_found","note":"Install with: npm install -g @google/gemini-cli"}'
  fi
}

# ── Helper: Extract JSON from response ───────────────────────
extract_json() {
  python3 - <<PYEOF
import sys, json, re
raw = sys.stdin.read()
# Strip markdown code fences
raw = re.sub(r'\x60\x60\x60json\s*', '', raw)
raw = re.sub(r'\x60\x60\x60\s*', '', raw)
# Find JSON object
match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', raw, re.DOTALL)
if not match:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
if match:
    try:
        data = json.loads(match.group())
        print(json.dumps(data))
        sys.exit(0)
    except:
        pass
# Fallback
print(json.dumps({
    "iteration": 0,
    "phase": "ERROR",
    "hypothesis": "Parse failed",
    "action_taken": "N/A",
    "result": raw[:300],
    "confidence": 0,
    "done": False,
    "next_action": "Retry with clearer prompt",
    "why_not_done": "Could not parse response"
}))
PYEOF
}

# ── MAIN OODA LOOP ────────────────────────────────────────────
for i in $(seq 1 "$MAX_ITER"); do

  echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}  🔁 ITERATION $i / $MAX_ITER${RESET}"
  echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

  # Read current state
  STATE=$(cat "$STATE_FILE")
  HISTORY=$(tail -n 80 "$MEMORY_FILE")

  # Every 5 iterations: inject meta-review
  META_INJECT=""
  if (( i % 5 == 0 && i > 1 )); then
    echo -e "${CYAN}🧠 Running meta-review...${RESET}"
    META_INJECT="CRITICAL META-REVIEW REQUIRED: You've run $i iterations. Question every assumption made so far. What are you doing wrong? Radically change strategy."
  fi

  # Build prompt
  PROMPT="You are MARK5, a relentless autonomous problem-solving agent in GODMODE/L99.

GOAL: $GOAL
ITERATION: $i of $MAX_ITER
$META_INJECT

MEMORY OF PREVIOUS ATTEMPTS:
$HISTORY

CURRENT STATE:
$STATE

INSTRUCTIONS:
- Analyze what has and hasn't worked
- Run the next OODA (Observe-Orient-Decide-Act) iteration
- Each attempt MUST be meaningfully different from previous ones
- If stuck, reframe the entire problem
- Be specific and concrete in actions

RESPOND WITH ONLY THIS JSON (no markdown, no explanation outside JSON):
{
  \"iteration\": $i,
  \"phase\": \"OBSERVE|ORIENT|DECIDE|ACT|VERIFY\",
  \"hypothesis\": \"your current best theory about what will work\",
  \"action_taken\": \"specific concrete action you are doing now\",
  \"result\": \"detailed result or output of the action\",
  \"confidence\": 0-100,
  \"done\": true or false,
  \"next_action\": \"if not done, what to try next\",
  \"why_not_done\": \"precise gap between current state and goal\",
  \"key_insight\": \"most important thing learned this iteration\"
}"

  echo -e "${CYAN}⚡ Calling Gemini...${RESET}"
  RAW_RESPONSE=$(call_gemini "$PROMPT")
  JSON_RESPONSE=$(echo "$RAW_RESPONSE" | extract_json)

  # Display parsed response
  echo ""
  echo "$JSON_RESPONSE" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f'  Phase      : {d.get(\"phase\",\"?\")}')
    print(f'  Hypothesis : {d.get(\"hypothesis\",\"?\")[:80]}')
    print(f'  Action     : {d.get(\"action_taken\",\"?\")[:80]}')
    print(f'  Result     : {d.get(\"result\",\"?\")[:80]}')
    print(f'  Insight    : {d.get(\"key_insight\",\"?\")[:80]}')
    print(f'  Confidence : {d.get(\"confidence\",0)}%')
    print(f'  Done       : {d.get(\"done\",False)}')
    if not d.get('done'):
        print(f'  Next       : {d.get(\"next_action\",\"?\")[:80]}')
        print(f'  Gap        : {d.get(\"why_not_done\",\"?\")[:80]}')
except Exception as e:
    print(f'  [Parse error: {e}]')
    print(sys.stdin.read()[:300])
" 2>/dev/null || echo "  [Response displayed above]"

  # Append to memory
  {
    echo ""
    echo "## Iteration $i — $(date +%H:%M:%S)"
    echo "\`\`\`json"
    echo "$JSON_RESPONSE"
    echo "\`\`\`"
  } >> "$MEMORY_FILE"

  # Update state with Python
  python3 - <<PYEOF
import json

with open("$STATE_FILE") as f:
    state = json.load(f)

try:
    new = json.loads('''$JSON_RESPONSE''')
    state["iteration"] = $i
    state["confidence"] = new.get("confidence", 0)
    state["done"] = new.get("done", False)
    
    # Track best attempt
    if new.get("confidence", 0) > state.get("best_confidence", 0):
        state["best_confidence"] = new.get("confidence", 0)
        state["best_attempt"] = new.get("result", "")
    
    # Track stuckness
    history = state.get("history", [])
    if len(history) >= 2 and history[-1].get("action") == new.get("action_taken"):
        state["stuck_count"] = state.get("stuck_count", 0) + 1
    else:
        state["stuck_count"] = 0
    
    history.append({
        "iteration": $i,
        "phase": new.get("phase",""),
        "action": new.get("action_taken","")[:100],
        "result": new.get("result","")[:100],
        "confidence": new.get("confidence", 0),
        "insight": new.get("key_insight","")[:100]
    })
    state["history"] = history[-15:]  # Keep last 15

except Exception as e:
    state["last_error"] = str(e)

with open("$STATE_FILE", "w") as f:
    json.dump(state, f, indent=2)
PYEOF

  # Check if done
  DONE_STATUS=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
done = s.get('done', False)
conf = s.get('confidence', 0)
stuck = s.get('stuck_count', 0)
print(f'{\"DONE\" if done and conf >= 90 else \"STUCK\" if stuck >= 5 else \"CONTINUE\"},{conf},{stuck}')
")
  
  STATUS=$(echo "$DONE_STATUS" | cut -d',' -f1)
  CONF=$(echo "$DONE_STATUS" | cut -d',' -f2)
  STUCK=$(echo "$DONE_STATUS" | cut -d',' -f3)

  if [ "$STATUS" = "DONE" ]; then
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║   ✅ MARK5: GOAL ACHIEVED!                            ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    echo -e "Completed in ${BOLD}$i iterations${RESET} with ${BOLD}$CONF% confidence${RESET}"
    
    # Generate final report
    FINAL_PROMPT="Summarize what was accomplished for this goal: '$GOAL'
    Based on this run history: $(tail -n 50 $MEMORY_FILE)
    Write a clean, actionable final report in markdown."
    
    call_gemini "$FINAL_PROMPT" > "$RESULT_FILE" 2>/dev/null || \
      echo "Goal achieved after $i iterations. See memory.md for full details." > "$RESULT_FILE"
    
    echo -e "${BOLD}📄 Final report: $RESULT_FILE${RESET}"
    echo -e "${BOLD}📁 Full logs: $LOG_DIR${RESET}"
    exit 0
  fi

  if [ "$STATUS" = "STUCK" ]; then
    echo -e "${RED}⚠️  STUCK detected ($STUCK repeated actions). Forcing reframe...${RESET}"
    # Force reframe by modifying memory
    echo "" >> "$MEMORY_FILE"
    echo "## ⚠️ REFRAME FORCED at iteration $i" >> "$MEMORY_FILE"
    echo "Previous approach failed. Completely change strategy." >> "$MEMORY_FILE"
  fi

  echo -e "${CYAN}⏳ Waiting ${DELAY}s before next iteration...${RESET}"
  sleep "$DELAY"

done

echo -e "${RED}Max iterations ($MAX_ITER) reached without achieving goal.${RESET}"
echo -e "Best attempt saved in: $LOG_DIR"
exit 1
