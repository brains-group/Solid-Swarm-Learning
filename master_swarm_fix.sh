#!/bin/bash

# --- CONFIGURATION ---
SESSION_NAME="swarm_orchestrator"
LOG_FILE="master_swarm.log"

# --- TMUX PERSISTENCE LAYER ---
# Check if we are already inside a tmux session
if [ -z "$TMUX" ]; then
    # Check if the session already exists
    tmux has-session -t "$SESSION_NAME" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "🔄 Swarm session '$SESSION_NAME' is already running. Re-attaching..."
        sleep 1
        exec tmux attach -t "$SESSION_NAME"
    else
        echo "🚀 No active session found. Launching new Tmux orchestrator..."
        echo "--- New Execution Started: $(date) ---" > "$LOG_FILE" # delete old log
        sleep 1
        # Create the session and run THIS script inside it
        tmux new-session -d -s "$SESSION_NAME" "bash \"$0\"; bash"
        exec tmux attach -t "$SESSION_NAME"
    fi
fi

# ---------------------------------------------------------------------------
# CORE ORCHESTRATION LOGIC (Now running safely inside Tmux)
# ---------------------------------------------------------------------------

# --- LOGGING SETUP ---
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🚀 Initiating the Master Swarm Orchestrator (with Auto-Reload)..."

# Extract Configuration from Python config
CONFIG_FILE="learning/data/src/config.py"
if [ -f "$CONFIG_FILE" ]; then
    # Uses grep/sed to find the line and extract the number
    NUM_CLIENTS=$(grep "NUM_ACTIVE_CLIENTS =" "$CONFIG_FILE" | sed 's/[^0-9]*//g')
    TOTAL_EPOCHS=$(grep "TOTAL_EPOCHS =" "$CONFIG_FILE" | sed 's/[^0-9]*//g')
    echo "📊 Configured for $NUM_CLIENTS active clients and $TOTAL_EPOCHS total epochs."
else
    echo "⚠️ Warning: config.py not found. Defaulting to 50 clients and 25 epochs."
    NUM_CLIENTS=50
    TOTAL_EPOCHS=25
fi

GLOBAL_DATA_DIR="learning/data/global"

# Determine latest epoch using Python to robustly pull digits out of filenames
LATEST_EPOCH=$(python3 - <<'PY'
import os, re
d = r'$GLOBAL_DATA_DIR'
m = 0
if os.path.exists(d):
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        # Only inspect regular files that look like global model weights (avoid logs/images/other files)
        if not os.path.isfile(fp):
            continue
        lf = f.lower()
        if not lf.endswith('.pt'):
            continue
        if 'global_model' not in lf and 'global' not in lf:
            continue
        # Prefer explicit _e<NUM>.pt pattern
        m1 = re.search(r'_e(\d+)\.pt$', lf)
        if m1:
            m = max(m, int(m1.group(1)))
            continue
        # Fallback: try to find any trailing number groups
        nums = re.findall(r'\d+', lf)
        if nums:
            m = max(m, max(map(int, nums)))
print(m)
PY
 2>/dev/null || echo 0)

if [ -z "$LATEST_EPOCH" ]; then
    LATEST_EPOCH=0
fi

# --- RESUME / RESTART LOGIC ---
RESUME_MODE=false
if [ "$LATEST_EPOCH" -gt 0 ]; then
    if [ "$LATEST_EPOCH" -ge "$TOTAL_EPOCHS" ]; then
        echo "✅ Final model weights for epoch $LATEST_EPOCH detected (Total expected: $TOTAL_EPOCHS)."
        echo "🔄 Restarting training from scratch..."
        RESUME_MODE=false
    else
        echo "⚠️ Crash detected! Found model weights up to epoch $LATEST_EPOCH."
        echo "🚀 Resuming training from where it left off..."
        RESUME_MODE=true
    fi
else
    echo "ℹ️ No existing model weights found. Starting fresh..."
    RESUME_MODE=false
fi

# 1. SETUP ENVIRONMENT
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 20 || nvm install 20

# Force standard environment variables for the Solid Server
export SOLID_SERVER_URL="http://localhost:3000"
export MY_SOLID_IDP="http://localhost:3000"

# Activate Python Virtual Env
VENV_PATH="$(pwd)/.env/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "✅ Virtual Environment Activated"
else
    echo "❌ ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Cleanup and Trap
trap 'echo -e "\n🛑 Shutting down background servers..."; kill $(jobs -p) 2>/dev/null; exit' EXIT

echo "🧹 Clearing ghost processes..."
lsof -ti:3000,3001,8545 | xargs kill -9 2>/dev/null || true
sleep 2

# Clean previous run state if restarting from scratch
if [ "$RESUME_MODE" = false ] && [ -d "$GLOBAL_DATA_DIR" ]; then
    echo "🧹 Wiping old global .pt models to ensure a clean start..."
    find "$GLOBAL_DATA_DIR" -maxdepth 1 -type f -name "*.pt" -exec rm -f {} + 2>/dev/null
fi

# --- PHASE 1: INFRASTRUCTURE ---

echo "▶️ [1/9] Booting Solid Server..."
cd solid_backend
if [ "$RESUME_MODE" = false ]; then
    echo "🧹 Removing old Solid pod data..."
    rm -rf my-solid-data .internal
else
    echo "♻️ Preserving Solid pod data for resume..."
fi
npm run build > /dev/null 2>&1
nvm exec 20 node --require cross-fetch/polyfill dist/SolidApplication.js > ../solid.log 2>&1 &
cd ..

echo "▶️ [2/9] Booting Anvil Blockchain..."
cd swarm_orchestrator
anvil --block-time 1 --gas-limit 3000000000 > ../anvil.log 2>&1 &
cd ..

# HEARTBEAT CHECK
echo "⏳ Waiting for Solid Server to respond on port 3000..."
MAX_RETRIES=6
COUNT=0
while ! curl -s http://localhost:3000 > /dev/null; do
    printf "."
    sleep 10
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo -e "\n❌ ERROR: Solid Server failed to boot. Check solid.log"
        exit 1
    fi
done
echo -e "\n✅ Solid Server is UP!"

# --- PHASE 2: SETUP ---

if [ "$RESUME_MODE" = false ]; then
    echo "▶️ [3/9] Generating Solid Accounts..."
    cd solid_backend/scripts
    node --require cross-fetch/polyfill --dns-result-order=ipv4first create_accounts.js
    cd ../..
else
    echo "▶️ [3/9] ♻️ Skipping Solid Account Generation (Resume Mode)..."
fi

echo "▶️ [4/9] Deploying Smart Contracts..."
cd swarm_orchestrator
forge script script/Deploy_SC.sol --rpc-url http://127.0.0.1:8545 --broadcast
cd ..

# --- PHASE 3: DATA & EXECUTION ---

cd learning/data/src

if [ "$RESUME_MODE" = false ]; then
    echo "▶️ [5/9] Partitioning Data..."
    python3 partition_data.py
else
    echo "▶️ [5/9] ♻️ Skipping Data Partitioning (Resume Mode)..."
fi

echo "▶️ [6/9] Registering Clients..."
python3 register_clients.py

echo "▶️ [7/9] Configuring Swarm Privacy, DP & Vulnerability Toggles..."
# Initializing nodes with a baseline Epsilon of 10.0
for i in $(seq 1 "$NUM_CLIENTS"); do
    python3 toggle_dp.py "$i" 10.0 > /dev/null 2>&1
    echo "✅ DP & Vulnerability values set for Pod $i"
done
echo "✅ Privacy baseline set for $NUM_CLIENTS nodes (Epsilon: 10.0)."

echo "▶️ [8/9] Launching Swarm Training Loop..."
# Export START_EPOCH so training processes can pick up where they left off
if [ "$RESUME_MODE" = true ]; then
    # Resume from the next epoch after the latest completed one
    NEXT_EPOCH=$((LATEST_EPOCH + 1))
    export START_EPOCH="$NEXT_EPOCH"
    echo "🔁 Resuming from epoch $START_EPOCH (latest complete was $LATEST_EPOCH)"
else
    export START_EPOCH=0
fi

python3 run_swarm.py

echo "▶️ [9/9] Training Complete! Evaluating Swarm Network..."
python3 evaluate_swarm.py

echo ""
echo "🎉 SWARM PIPELINE FULLY EXECUTED! 🎉"
echo "--------------------------------------------------------"
echo "TIP: You can detach from this view by pressing Ctrl+B, then D."
echo "     Re-run this script later to check progress."
echo "--------------------------------------------------------"

# Wait for background jobs (Solid, Anvil, Monitor) to keep session alive
wait