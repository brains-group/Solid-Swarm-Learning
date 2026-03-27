# Global Configuration for Swarm Learning Project

NUM_ACTIVE_CLIENTS = 15
TOTAL_EPOCHS = 30

# --- Top-N Recommendation Settings ---
TOP_N = 10 
NUM_NEGATIVE_SAMPLES = 99 # We mix 1 "True" recipe with 99 "Fake" ones for evaluation
TOTAL_RECIPES = 267783 # The exact max ID of your dataset