# Global Configuration for Swarm Learning Project

NUM_ACTIVE_CLIENTS = 100
TOTAL_EPOCHS = 50
ROUNDS_PER_EPOCH = 3

# Minimum number of reviews a reviewer must have to be eligible as a client
MIN_REVIEWS_PER_CLIENT = 500

# --- Top-N Recommendation Settings ---
TOP_N = 10 
NUM_NEGATIVE_SAMPLES = 99 # We mix 1 "True" recipe with 99 "Fake" ones for evaluation
TOTAL_RECIPES = 500000 # The exact max ID of your dataset

# Maximum concurrent training processes to avoid OOM. Tune to available RAM.
MAX_CONCURRENT_CLIENTS = 15

#size of batches used during ML. The bigger the matrix, the faster the training (up to a point), but also the more RAM is needed. Tune to available RAM.
BATCH_SIZE_GLOBAL = 512
#size of embeddings for each item. Higher means more capacity but also more RAM usage and slower speed. Tune to available RAM and dataset complexity.
EMBEDDING_DIM = 128
