# Global Configuration for Swarm Learning Project

NUM_ACTIVE_CLIENTS = 10
TOTAL_EPOCHS = 20

# --- Top-N Recommendation Settings ---
TOP_N = 10 
NUM_NEGATIVE_SAMPLES = 99 # We mix 1 "True" recipe with 99 "Fake" ones for evaluation
TOTAL_RECIPES = 500000 # The exact max ID of your dataset