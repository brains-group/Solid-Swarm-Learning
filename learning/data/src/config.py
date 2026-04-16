# Global Configuration for Swarm Learning Project

NUM_ACTIVE_CLIENTS = 150
TOTAL_EPOCHS = 50
ROUNDS_PER_EPOCH = 1

# Minimum number of reviews a reviewer must have to be eligible as a client
MIN_REVIEWS_PER_CLIENT = 250

# --- Top-N Recommendation Settings ---
TOP_N = 10 
NUM_NEGATIVE_SAMPLES = 99 # We mix 1 "True" recipe with 99 "Fake" ones for evaluation
TOTAL_RECIPES = 500000 # The exact max ID of your dataset

# Maximum concurrent training processes to avoid OOM. Tune to available RAM.
MAX_CONCURRENT_CLIENTS = 15

