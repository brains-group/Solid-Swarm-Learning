import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from config import NUM_ACTIVE_CLIENTS, MIN_REVIEWS_PER_CLIENT
from solid_integration import SolidTokenClient

# Configuration
PODS_DIR = None

def setup_directories(client_id):
    """Creates the isolated directory structure for a client."""
    path = os.path.join(PODS_DIR, f'client_{client_id}')
    os.makedirs(path, exist_ok=True)
    return path

def assign_swarm_by_genres(user_interactions):
    """Simple swarm assignment: for now all clients go to swarm 0.
    Could be extended to use dominant genres.
    """
    return {
        "swarm_ids": [0],
        "genres": user_interactions['genres'].explode().value_counts().head(3).to_dict()
    }

def main():
    # 1. LOAD THE MASTER CREDENTIALS FILE
    # Resolve path relative to this script to be robust to current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    accounts_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'secret', 'user_accounts.json'))
    try:
        with open(accounts_path, 'r') as f:
            user_accounts = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find {accounts_path}. Please ensure the Node.js backend credentials exist.")
        return

    # Ensure pods dir is resolved relative to the script
    global PODS_DIR
    PODS_DIR = os.path.abspath(os.path.join(script_dir, '..', 'pods'))

    print("Loading MovieLens ml-32m datasets...")
    # Resolve the global data directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'global', 'ml-32m'))
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    movies_path = os.path.join(data_dir, 'movies.csv')

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        print(f"❌ Error: Expected files not found in {data_dir}. Ensure 'ratings.csv' and 'movies.csv' exist.")
        return

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Normalize column names
    ratings.rename(columns={
        'userId': 'user_id',
        'movieId': 'movie_id',
        'rating': 'rating'
    }, inplace=True)

    movies.rename(columns={
        'movieId': 'movie_id',
        'title': 'title',
        'genres': 'genres'
    }, inplace=True)

    # Merge ratings with movie metadata
    interactions = ratings.merge(movies, on='movie_id', how='left')

    # Convert genres to list for easier processing
    interactions['genres'] = interactions['genres'].fillna('').apply(lambda g: g.split('|') if g else [])

    # Binary label: positive if rating > 3
    interactions['label'] = (interactions['rating'] > 3).astype(int)

    print("Filtering for eligible reviewers...")
    user_counts = interactions['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= MIN_REVIEWS_PER_CLIENT].index.tolist()

    if len(eligible_users) == 0:
        print(f"❌ No reviewers have >= {MIN_REVIEWS_PER_CLIENT} reviews. Lower MIN_REVIEWS_PER_CLIENT in config.py.")
        return

    # Randomly select users to act as clients
    if len(eligible_users) < NUM_ACTIVE_CLIENTS:
        print(f"⚠️ Only {len(eligible_users)} eligible reviewers available, but NUM_ACTIVE_CLIENTS={NUM_ACTIVE_CLIENTS}. Reducing client count.")
        selected_users = eligible_users
    else:
        random.seed(42)
        selected_users = random.sample(eligible_users, NUM_ACTIVE_CLIENTS)

    # Enforce the minimum review threshold defensively in case of unexpected filtering issues
    filtered_selected = [u for u in selected_users if user_counts.get(u, 0) >= MIN_REVIEWS_PER_CLIENT]
    if len(filtered_selected) != len(selected_users):
        print(f"⚠️ Filtered out {len(selected_users) - len(filtered_selected)} selected users who didn't meet MIN_REVIEWS_PER_CLIENT={MIN_REVIEWS_PER_CLIENT}.")
        selected_users = filtered_selected

    if len(selected_users) == 0:
        print(f"❌ No users left after enforcing MIN_REVIEWS_PER_CLIENT={MIN_REVIEWS_PER_CLIENT}. Aborting.")
        return

    print(f"Partitioning data and provisioning Solid Pods for {len(selected_users)} clients...")
    swarm_id_counts = {}

    for idx, user_id in enumerate(selected_users):
        client_num = idx + 1
        client_dir = setup_directories(client_num)
        
        # 2. MATCH PYTHON CLIENT TO SOLID USER (client_1 -> user1)
        solid_user_key = f"user{client_num}"
        if solid_user_key not in user_accounts:
            print(f"⚠️ Warning: No credentials found for {solid_user_key} in JSON. Skipping.")
            continue
            
        credentials = user_accounts[solid_user_key]
        
        # Isolate and split the user's data
        user_data = interactions[interactions['user_id'] == user_id].copy()
        # Keep only relevant columns for training/upload
        user_data = user_data[['movie_id', 'title', 'genres', 'rating', 'label']]

        train_df, test_df = train_test_split(user_data, test_size=0.2, random_state=42)
        train_common, train_vulnerable = train_test_split(train_df, test_size=0.3, random_state=42)

        # Assign to swarms (simple default: single swarm)
        profile = assign_swarm_by_genres(user_data)
        for swarm_id in profile.get('swarm_ids', [0]):
            swarm_id_counts[swarm_id] = swarm_id_counts.get(swarm_id, 0) + 1
        
        # 3. INJECT SOLID CREDENTIALS INTO THE LOCAL PROFILE
        profile['solid_pod_url'] = credentials['pod'].replace(f"/{solid_user_key}/", "") # e.g., http://localhost:3000
        profile['solid_username'] = solid_user_key
        profile['solid_token_id'] = credentials['client_credentials_token_identifier']
        profile['solid_token_secret'] = credentials['client_credentials_token_secret']
        
        # Save locally first
        train_common_path = os.path.join(client_dir, 'train_common.csv')
        train_vuln_path = os.path.join(client_dir, 'train_vulnerable.csv')
        test_path = os.path.join(client_dir, 'test.csv')

        # Save CSVs; genres lists are JSON-serializable, convert to pipe-delimited for CSV ease
        def prepare_df(df):
            df_out = df.copy()
            # Normalize possible legacy column names
            if 'userId' in df_out.columns or 'movieId' in df_out.columns:
                df_out = df_out.rename(columns={
                    'userId': 'user_id',
                    'movieId': 'movie_id'
                })

            # Ensure genres column exists and is pipe-delimited
            if 'genres' in df_out.columns:
                df_out['genres'] = df_out['genres'].apply(lambda g: '|'.join(g) if isinstance(g, list) else (g if isinstance(g, str) else ''))
            else:
                df_out['genres'] = ''

            # Ensure title exists
            if 'title' not in df_out.columns:
                df_out['title'] = ''

            # Ensure label exists
            if 'label' not in df_out.columns and 'rating' in df_out.columns:
                df_out['label'] = (df_out['rating'] > 3).astype(int)

            # Add recipe_id column for compatibility with existing training code
            if 'movie_id' in df_out.columns:
                df_out['recipe_id'] = df_out['movie_id']

            # Force column order and drop extras
            desired = ['recipe_id', 'movie_id', 'title', 'genres', 'rating', 'label']
            for c in desired:
                if c not in df_out.columns:
                    df_out[c] = ''

            return df_out[desired]

        common_out = prepare_df(train_common)
        vuln_out = prepare_df(train_vulnerable)
        test_out = prepare_df(test_df)

        common_out.to_csv(train_common_path, index=False)
        vuln_out.to_csv(train_vuln_path, index=False)
        test_out.to_csv(test_path, index=False)

        # Print how many reviews this client received (show original total and training split)
        total_original = len(train_common) + len(train_vulnerable) + len(test_df)
        num_reviews = len(common_out) + len(vuln_out)
        print(f"✓ Client {client_num} receives {num_reviews} training reviews (common={len(common_out)}, vulnerable={len(vuln_out)}); original total={total_original}")
        
        with open(os.path.join(client_dir, 'profile.json'), 'w') as f:
            json.dump(profile, f, indent=4)
            
        # 4. UPLOAD TO THE REAL SOLID POD
        print(f"\nProvisioning Solid Pod for {solid_user_key}...")
        solid_client = SolidTokenClient(
            pod_url=profile['solid_pod_url'],
            username=profile['solid_username'],
            token_id=profile['solid_token_id'],
            token_secret=profile['solid_token_secret']
        )
        
        if solid_client.authenticate():
            solid_client.create_container("/swarm_data/")
            solid_client.upload_file(train_common_path, "/swarm_data/train_common.csv")
            solid_client.upload_file(train_vuln_path, "/swarm_data/train_vulnerable.csv")
            solid_client.upload_file(test_path, "/swarm_data/test.csv")
        
        print(f"✓ Client {client_num} fully provisioned.")

    print("\n--- Swarm Assignment Summary ---")
    # Sort by swarm ID for consistent output
    sorted_counts = sorted(swarm_id_counts.items())

    for swarm_id, count in sorted_counts:
        print(f"Swarm ID {swarm_id}: {count} clients")

    # Total reviews across all selected clients
    total_reviews = 0
    for idx, user_id in enumerate(selected_users):
        client_dir = setup_directories(idx+1)
        common_path = os.path.join(client_dir, 'train_common.csv')
        vuln_path = os.path.join(client_dir, 'train_vulnerable.csv')
        c = 0
        if os.path.exists(common_path):
            c += sum(1 for _ in open(common_path)) - 1
        if os.path.exists(vuln_path):
            c += sum(1 for _ in open(vuln_path)) - 1
        print(f"Client {idx+1} total reviews: {c}")
        total_reviews += c

    print(f"Total reviews across all clients: {total_reviews}")
    print("---------------------------------")

if __name__ == "__main__":
    main()