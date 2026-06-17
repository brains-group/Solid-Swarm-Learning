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
        # Create or reset the log and run THIS script inside the session
        echo "--- New Execution Started: $(date) ---" > "$LOG_FILE"
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

# Ensure we only remove global .pt models (leave other data alone)
GLOBAL_DATA_DIR="learning/data/global"
if [ -d "$GLOBAL_DATA_DIR" ]; then
    echo "🧹 Cleaning any existing global .pt models..."
    find "$GLOBAL_DATA_DIR" -maxdepth 1 -type f -name "*.pt" -exec rm -f {} + 2>/dev/null || true
fi

# Wipe old local client weights to ensure a completely fresh start
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
lsof -ti:3000,3001,3333,8545 | xargs kill -9 2>/dev/null || true
sleep 2

# --- PHASE 1: INFRASTRUCTURE ---

echo "▶️ [1/9] Booting Solid Server..."
cd solid_backend
# THE FIX: Added 'data', '*.sqlite', and '*.db' to ensure total amnesia
rm -rf my-solid-data .internal data *.sqlite *.db
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


echo "▶️ [2.5/9] Starting PodOS environment..."

# 1. Load NVM
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

ORIGINAL_DIR=$(pwd)
cd PodOS || { echo "Error: PodOS folder not found!"; exit 1; }

nvm install 20 > /dev/null 2>&1
nvm use 20 > /dev/null 2>&1

hash -r
unalias npm 2>/dev/null
unalias node 2>/dev/null

NODE_BIN=$(nvm which 20)
NODE_DIR=$(dirname "$NODE_BIN")
export PATH="$NODE_DIR:$PATH"

# 2. Delete old logs ONLY (Removed the rm -rf deep clean!)
rm -f core.log elements.log install.log

echo "Installing PodOS dependencies (silently)..."
env CI=true npm ci --no-progress --loglevel=error > install.log 2>&1

# 3. Boot Core
echo "Booting PodOS core..."
npm run dev:core < /dev/null > core.log 2>&1 &

echo "⏳ Waiting for PodOS core to finish building..."
MAX_WAIT=60
WAIT_COUNT=0
while ! grep -q "build finished" core.log 2>/dev/null; do
    printf "."
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT+2))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo -e "\n❌ ERROR: PodOS core timeout. Check core.log"
        exit 1
    fi
done
echo -e "\n✅ PodOS core built successfully!"

# 4. Boot Elements (UI)
echo "Booting PodOS elements (UI)..."
npm run dev:elements < /dev/null > elements.log 2>&1 &

# 5. THE HTTP 200 SNIPER
# This holds the script until the server explicitly proves the JS file is ready!
echo "⏳ Waiting for StencilJS routing engine to mount JS bundles..."
WAIT_COUNT=0
while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:3333/build/pos-app-browser.entry.js) -ne 200 ]; do
    printf "."
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT+2))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo -e "\n❌ ERROR: PodOS UI timeout. The JS bundles never mounted."
        exit 1
    fi
done

echo -e "\n✅ PodOS UI Javascript bundles are fully mapped and serving!"

cd "$ORIGINAL_DIR"

# end of PodOS setup

echo "▶️ [3/9] Generating Solid Accounts..."
cd solid_backend/scripts
node --require cross-fetch/polyfill --dns-result-order=ipv4first create_accounts.js
cd ../..

# --- NEW STEP 3.5 ---
echo "▶️ [3.5/9] Attempting to open user1's Pod in an Incognito/Private Browser..."

# URL-encoded target URI: http://localhost:3000/user1/
USER1_POD_URI="http%3A%2F%2Flocalhost%3A3000%2Fuser1%2F"
PODOS_TARGET_URL="http://localhost:3333/?uri=$USER1_POD_URI"

# Helper function to quietly attempt running absolute paths
try_path() {
    "$@" > /dev/null 2>&1
    return $?
}

if [[ "$OSTYPE" == "darwin"* ]]; then
    # --- macOS ---
    # Try generic application calls first
    open -na "Google Chrome" --args --incognito "$PODOS_TARGET_URL" 2>/dev/null || \
    open -na "Firefox" --args --private-window "$PODOS_TARGET_URL" 2>/dev/null || \
    open -na "Brave Browser" --args --incognito "$PODOS_TARGET_URL" 2>/dev/null || \
    open -na "Microsoft Edge" --args --inprivate "$PODOS_TARGET_URL" 2>/dev/null || \
    # Fallback to absolute application paths
    try_path /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --incognito "$PODOS_TARGET_URL" || \
    try_path /Applications/Firefox.app/Contents/MacOS/firefox --private-window "$PODOS_TARGET_URL" || \
    # Ultimate fallback
    open "$PODOS_TARGET_URL"

elif [[ -n "$WSL_DISTRO_NAME" || "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # --- Windows (Native or via WSL) ---
    # Try Windows command aliases first
    cmd.exe /c start chrome --incognito "$PODOS_TARGET_URL" 2>/dev/null || \
    cmd.exe /c start firefox -private-window "$PODOS_TARGET_URL" 2>/dev/null || \
    cmd.exe /c start msedge -inprivate "$PODOS_TARGET_URL" 2>/dev/null || \
    # Fallback to common absolute Windows paths (useful if WSL aliases fail)
    try_path "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --incognito "$PODOS_TARGET_URL" || \
    try_path "/mnt/c/Program Files/Mozilla Firefox/firefox.exe" -private-window "$PODOS_TARGET_URL" || \
    try_path "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" -inprivate "$PODOS_TARGET_URL" || \
    # Ultimate fallback
    explorer.exe "$PODOS_TARGET_URL"

else
    # --- Native Linux ---
    # Check for specific binaries and launch them silently in the background
    if command -v google-chrome-stable >/dev/null 2>&1; then
        google-chrome-stable --incognito "$PODOS_TARGET_URL" &
    elif command -v google-chrome >/dev/null 2>&1; then
        google-chrome --incognito "$PODOS_TARGET_URL" &
    elif command -v firefox >/dev/null 2>&1; then
        firefox --private-window "$PODOS_TARGET_URL" &
    elif command -v brave-browser >/dev/null 2>&1; then
        brave-browser --incognito "$PODOS_TARGET_URL" &
    elif command -v chromium-browser >/dev/null 2>&1; then
        chromium-browser --incognito "$PODOS_TARGET_URL" &
    elif command -v chromium >/dev/null 2>&1; then
        chromium --incognito "$PODOS_TARGET_URL" &
    elif command -v xdg-open >/dev/null 2>&1; then
        echo "⚠️ Could not find a specific browser for incognito mode. Falling back to standard tab."
        xdg-open "$PODOS_TARGET_URL" &
    else
        echo "❌ No browser launch command found. Please open manually: $PODOS_TARGET_URL"
    fi
fi
# --------------------

echo "▶️ [4/9] Deploying Smart Contracts..."
cd swarm_orchestrator
forge script script/Deploy_SC.sol --rpc-url http://127.0.0.1:8545 --broadcast
cd ..

# --- PHASE 3: DATA & EXECUTION ---

cd learning/data/src

echo "▶️ [5/9] Partitioning Data..."
python3 partition_data.py
echo "▶️ [6/9] Registering Clients..."
python3 register_clients.py

echo "▶️ [7/9] Configuring Swarm Privacy, DP & Vulnerability Toggles..."
# Initializing 50 nodes with a baseline Epsilon of 10.0
# This registers the privacy state on the blockchain for each Pod
for i in $(seq 1 "$NUM_CLIENTS"); do
    python3 toggle_dp.py "$i" 10.0 > /dev/null 2>&1
    # Optional: toggle_vulnerability.py "$i" 0 (if you have a toggle script for vulnerability)
done
echo "✅ Privacy baseline set for $NUM_CLIENTS nodes (Epsilon: 10.0)."

echo "▶️ [8/9] Launching Swarm Training Loop..."
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