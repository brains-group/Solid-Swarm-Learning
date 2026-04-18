#!/bin/bash
# Force the system to allow a larger backlog of connections
ulimit -n 10000
ulimit -u 10000


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

echo "🚀 Initiating the True Master Swarm Orchestrator..."

# Wipe old global models for a fresh start
GLOBAL_DATA_DIR="learning/data/global"
if [ -d "$GLOBAL_DATA_DIR" ]; then
    echo "🧹 Cleaning any existing global .pt models..."
    find "$GLOBAL_DATA_DIR" -maxdepth 1 -type f -name "*.pt" -exec rm -f {} + 2>/dev/null || true
fi

# Wipe old local client weights
PODS_DATA_DIR="learning/data/pods"
if [ -d "$PODS_DATA_DIR" ]; then
    echo "🧹 Cleaning any existing local client .pt models..."
    find "$PODS_DATA_DIR" -type f -name "*.pt" -exec rm -f {} + 2>/dev/null || true
fi

# Extract Number of Clients from Python config
CONFIG_FILE="learning/data/src/config.py"
if [ -f "$CONFIG_FILE" ]; then
    # Uses grep/sed to find the line and extract the number
    NUM_CLIENTS=$(grep "NUM_ACTIVE_CLIENTS =" "$CONFIG_FILE" | sed 's/[^0-9]*//g')
    echo "📊 Configured for $NUM_CLIENTS active clients."
else
    echo "⚠️ Warning: config.py not found. Defaulting to 50 clients."
    NUM_CLIENTS=50
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
# Note: In Tmux, killing the foreground process won't necessarily close the window
trap 'echo -e "\n🛑 Shutting down background servers..."; kill $(jobs -p) 2>/dev/null; exit' EXIT

echo "🧹 Clearing ghost processes..."
lsof -ti:3000,3001,8545 | xargs kill -9 2>/dev/null || true
sleep 2

# --- PHASE 1: INFRASTRUCTURE ---

echo "▶️ [1/7] Booting Solid Server..."
cd solid_backend
npm run build > /dev/null 2>&1
nvm exec 20 node --require cross-fetch/polyfill dist/SolidApplication.js > ../solid.log 2>&1 &
cd ..

echo "▶️ [2/7] Booting Anvil Blockchain..."
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

echo "▶️ [3/7] Deploying Smart Contracts..."
cd swarm_orchestrator
forge script script/Deploy_SC.sol --rpc-url http://127.0.0.1:8545 --broadcast
cd ..

# --- PHASE 3: DATA & EXECUTION ---

cd learning/data/src

echo "▶️ [4/7] Registering Clients..."
python3 register_clients.py

echo "▶️ [5/7] Configuring Swarm Privacy, DP & Vulnerability Toggles..."
# Initializing 50 nodes with a baseline Epsilon of 10.0
# This registers the privacy state on the blockchain for each Pod
for i in $(seq 1 "$NUM_CLIENTS"); do
    python3 toggle_dp.py "$i" 10.0 > /dev/null 2>&1
    if (( i % 2 == 0 )); then
        python3 toggle_privacy.py "$i" exclude > /dev/null 2>&1
    fi
    echo "✅ DP & Vulnerability values set for Pod $i"
    # Optional: toggle_vulnerability.py "$i" 0 (if you have a toggle script for vulnerability)
done
echo "✅✅ Privacy baseline set for $NUM_CLIENTS nodes (Epsilon: 10.0)."

echo "▶️ [6/7] Launching Swarm Training Loop..."
python3 run_swarm.py

echo "▶️ [7/7] Training Complete! Evaluating Swarm Network..."
python3 evaluate_swarm.py

echo ""
echo "🎉 SWARM PIPELINE FULLY EXECUTED! 🎉"
echo "--------------------------------------------------------"
echo "TIP: You can detach from this view by pressing Ctrl+B, then D."
echo "     Re-run this script later to check progress."
echo "--------------------------------------------------------"

# Wait for background jobs (Solid, Anvil, Monitor) to keep session alive
wait