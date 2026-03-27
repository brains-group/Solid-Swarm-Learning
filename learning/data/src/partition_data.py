import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import re
from config import NUM_ACTIVE_CLIENTS

# Configuration
MIN_INTERACTIONS = 20
MAX_TASTE_PROFILES = 2
GLOBAL_DATA_DIR = '../global'
PODS_DIR = '../pods'
TASTE_TO_SWARM_ID = {
    'sweet': 0, 'savory': 1, 'sour': 2, 'salty': 3, 'bitter': 4
}

def setup_directories(client_id):
    """Creates the isolated directory structure for a client."""
    path = os.path.join(PODS_DIR, f'client_{client_id}')
    os.makedirs(path, exist_ok=True)
    return path

def extract_taste_profile(user_interactions, recipes_df, user_id, ingredient_map):
    """
    Generates a mock taste profile based on the recipes the user interacted with.
    This version uses recipe ingredient tokens (after mapping IDs to names) to categorize users
    into up to 3 taste profiles (sweet, savory, sour, salty, bitter) if their preferences are mixed.
    """
    # Merge user interactions with recipe metadata
    merged = user_interactions.merge(recipes_df, left_on='recipe_id', right_on='id')
    
    # Define keywords for major taste profiles
    # Expanded keywords for major taste profiles based on dataset extraction
    TASTE_KEYWORDS = {
        'sweet': {
            'sugar', 'honey', 'syrup', 'sweet', 'candy', 'molasses', 'stevia',
            'sucralose', 'splenda', 'confectioner'
        },
        
        'savory': {
            'umami', 'mushroom', 'parmesan', 'yeast', 'broth', 'stock', 'sausage',
            'lamb', 'veal', 'steak', 'burger', 'chili', 'gravy', 'bouillon',
            'shallot', 'scallion', 'leek', 'chive', 'meatball', 'sesame', 'tahini',
            'oregano', 'thyme', 'rosemary', 'marjoram', 'tarragon', 'cumin', 'coriander',
            'paprika', 'bay leaf', 'onion', 'garlic', 'tomato'
        },
        
        'sour': {
            # --- Citrus & Core Acids ---
            'lemon', 'lime', 'citrus', 'grapefruit', 'orange', 'tangerine', 'clementine', 
            'pomelo', 'yuzu', 'calamansi', 'kumquat', 'citric', 'tartar', 'malic', 'lactic',

            # --- Vinegars & Fermentations ---
            'vinegar', 'balsamic', 'cider', 'fermented', 'pickled', 'pickle', 'kombucha', 
            'kimchi', 'kraut', 'sourdough', 'escabeche', 'cornichon', 'caper', 'relish',

            # --- Cultured Dairy ---
            'yogurt', 'buttermilk', 'kefir', 'sour cream', 'cream cheese', 'goat cheese', 
            'feta', 'blue cheese', 'creme fraiche', 'labneh', 'tzatziki', 'quark',

            # --- Tart Fruits & Vegetables ---
            'tomato', 'tomatillo', 'cranberry', 'rhubarb', 'gooseberry', 'plum', 'kiwi', 
            'pomegranate', 'passionfruit', 'passion fruit', 'barberry', 'quince', 'apple', 
            'applesauce', 'cherry', 'umeboshi', 

            # --- Tart Condiments & Sauces ---
            'mustard', 'mayonnaise', 'ketchup', 'ponzu', 'chutney', 'ranch', 'verjus', 

            # --- Herbs, Plants & Tart Spices ---
            'tamarind', 'hibiscus', 'sumac', 'sorrel', 'dill', 'lemongrass', 'amchur', 
            'tajin', 'chamoy', 'roselle', 'sour'
        },
        
        'salty': {
            # --- Base Salts, Seasonings & Mixes ---
            'salt', 'sea salt', 'kosher', 'msg', 'celery salt', 'garlic salt', 'onion salt', 
            'seasoning', 'old bay', 'taco seasoning', 'cajun seasoning', 'creole seasoning',

            # --- Sauces, Pastes & Condiments ---
            'soy', 'soy sauce', 'tamari', 'aminos', 'miso', 'teriyaki', 'hoisin',
            'ketchup', 'gravy', 'tapenade',

            # --- Pickled, Brined & Marine ---
            'brine', 'capers', 'caper', 'olives', 'olive', 'pickle', 
            'anchovy', 'sardine', 'caviar', 'roe', 'kelp', 'seaweed', 'dulse', 'nori',

            # --- Broths & Extract Condiments ---
            'broth', 'stock', 'bouillon', 'bouillon cube', 'vegemite', 'marmite', 
            'soup mix', 'onion soup mix', 

            # --- Cured/Processed Meats ---
            'bacon', 'prosciutto', 'pancetta', 'salami', 'pepperoni', 'ham', 'pastrami', 
            'corned', 'jerky', 'beef jerky', 'sausage', 'hot dog', 'bologna', 'spam',

            # --- Cheeses (Required for bare-word matching) ---
            'cheese', 'feta', 'halloumi', 'pecorino', 'romano', 'cotija', 'queso', 
            'cheddar', 'parmesan', 'asiago', 'blue cheese', 'mozzarella', 'gruyere', 
            'provolone', 'ricotta', 'gouda', 'brie', 'camembert', 'gorgonzola', 
            'fontina', 'havarti', 'mascarpone', 'muenster', 'roquefort', 'edam',

            # --- Salty Snacks, Chips & Junk Food ---
            'pretzel', 'cracker', 'popcorn', 'ramen', 'pork rinds', 'saltine', 'chip'
            'nacho', 'salted', 'peanut'
        },
        
        'bitter': { 
            'kale', 'radicchio', 'endive', 'arugula', 'dandelion', 'gourd', 'rucola', 'rocket',
            'frisee', 'escarole', 'treviso', 'horseradish', 'wasabi', 'fenugreek',
            'brussels sprout', 'collard', 'chard', 'watercress', 'artichoke', 'coffee',
            'espresso', 'matcha', 'chicory', 'cocoa', 'beer', 'tonic', 'hops',
            'bitters', 'campari', 'aperol', 'amaro', 'fernet', 'vermouth', 'absinthe',
            'chartreuse', 'juniper', 'gin', 'radish', 'eggplant', 'broccoli',
            'cabbage', 'chocolate', 'red wine', 'white wine'
        }
    }

    # --- Performance Optimization ---
    # Pre-compile regex for single-word keywords. Compiling regex is expensive,
    # so we do it once outside the main loops instead of on every check.
    COMPILED_TASTE_REGEX = {taste: [] for taste in TASTE_KEYWORDS}
    MULTI_WORD_KEYWORDS = {taste: [] for taste in TASTE_KEYWORDS}

    for taste, keywords in TASTE_KEYWORDS.items():
        for keyword in keywords:
            if ' ' in keyword:
                # Multi-word keywords can be checked with a fast 'in' operation.
                MULTI_WORD_KEYWORDS[taste].append(keyword)
            else:
                # Single-word keywords are compiled into a regex for whole-word matching.
                COMPILED_TASTE_REGEX[taste].append(re.compile(r'\b' + re.escape(keyword) + r'\b'))

    taste_scores = {taste: 0 for taste in TASTE_KEYWORDS}

    if 'ingredient_tokens' in merged.columns:
        for ingredients_str in merged['ingredient_tokens']:
            try:
                # The tokens are stored as a string representation of a list of IDs, e.g., "[123, 456]"
                ingredients_id_list = ast.literal_eval(ingredients_str)
                if not isinstance(ingredients_id_list, list):
                    continue
                
                # Flatten the list in case of nested lists of tokens, which can occur in the dataset.
                # E.g., "[123, [456, 789]]" -> [123, 456, 789]
                flat_ingredient_ids = []
                q = list(ingredients_id_list)
                while q:
                    item = q.pop(0)
                    if isinstance(item, list):
                        q = item + q # Prepend nested list items to the queue
                    else:
                        flat_ingredient_ids.append(item)
                
                # Map ingredient IDs to names
                ingredient_names = [ingredient_map.get(id, "") for id in flat_ingredient_ids]
                
                # For each taste profile, check if *any* ingredient in the recipe matches.
                # This ensures each recipe contributes at most +1 to any given taste's score,
                # preventing recipes with multiple similar ingredients from skewing the results.
                for taste in TASTE_KEYWORDS:
                    dish_matches_taste = False
                    for ing_name in ingredient_names:
                        # First, check for faster multi-word substring matches.
                        for keyword in MULTI_WORD_KEYWORDS[taste]:
                            if keyword in ing_name:
                                dish_matches_taste = True
                                break
                        if dish_matches_taste:
                            break
                        
                        # If no multi-word match, check the pre-compiled regex for single words.
                        for pattern in COMPILED_TASTE_REGEX[taste]:
                            if pattern.search(ing_name):
                                dish_matches_taste = True
                                break
                        if dish_matches_taste:
                            break
                    
                    if dish_matches_taste:
                        taste_scores[taste] += 1

            except (ValueError, SyntaxError):
                # Handle cases where the string is not a valid literal or is malformed
                continue
    
    total_interactions = len(user_interactions)
    if total_interactions == 0:
        return {
            "swarm_ids": [0],  # Default to savory
            "taste_profiles": [{"taste": "savory", "score": 0.0}],
            "interaction_count": 0,
            "debug_taste_scores": taste_scores
        }

    # Sort tastes by raw score in descending order to find the user's dominant profile(s)
    sorted_tastes = sorted(taste_scores.items(), key=lambda item: item[1], reverse=True)
    
    assigned_tastes = []
    if sorted_tastes and sorted_tastes[0][1] > 0:
        max_score = sorted_tastes[0][1]
        # A taste is included if its score is at least 75% of the top taste's score.
        RELATIVE_THRESHOLD = 0.75
        
        for taste, score in sorted_tastes:
            if score >= max_score * RELATIVE_THRESHOLD:
                # For the final output, we still use the score relative to total interactions,
                # as it's an intuitive measure of how often that taste appears.
                final_score = score / total_interactions
                assigned_tastes.append({"taste": taste, "score": round(final_score, 3)})
    
    # Limit to a maximum number of groups and handle the default case if no tastes were assigned
    assigned_tastes = assigned_tastes[:MAX_TASTE_PROFILES]
    if not assigned_tastes:
        assigned_tastes = [{"taste": "savory", "score": 0.0}]

    # All clients will be assigned to a single global swarm (ID 0)
    # to create one unified model.
    swarm_ids = [0]

    return {
        "swarm_ids": swarm_ids,
        "taste_profiles": assigned_tastes,
        "interaction_count": total_interactions,
        "debug_taste_scores": taste_scores # Useful to see why they got placed
    }

def main():
    print("Loading global datasets...")
    # Assuming you downloaded RAW_interactions.csv and PP_recipes.csv
    interactions = pd.read_csv(os.path.join(GLOBAL_DATA_DIR, 'RAW_interactions.csv'))
    recipes = pd.read_csv(os.path.join(GLOBAL_DATA_DIR, 'PP_recipes.csv'))

    print("Loading and processing ingredient map...")
    # The provided ingr_map.csv contains detailed ingredient mappings.
    # We need to extract a simple ID -> standardized name mapping from it.
    ingr_map_df = pd.read_csv(os.path.join(GLOBAL_DATA_DIR, 'ingr_map.csv'))
    
    # The relevant columns are 'id' (the ingredient ID) and 'replaced' (the standardized name).
    # We drop duplicates on 'id' to ensure a unique mapping from ID to name.
    ingr_map_clean = ingr_map_df[['id', 'replaced']].dropna().drop_duplicates(subset=['id'])
    ingredient_map = pd.Series(ingr_map_clean.replaced.values, index=ingr_map_clean.id).to_dict()
    
    print("Filtering for active users...")
    user_counts = interactions['user_id'].value_counts()
    valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index.tolist()

    # Limit to our simulation size
    selected_users = valid_users[:NUM_ACTIVE_CLIENTS]

    print(f"Partitioning data for {NUM_ACTIVE_CLIENTS} clients based on mixed taste profiles...")
    swarm_id_counts = {}
    for idx, user_id in enumerate(selected_users):
        client_dir = setup_directories(idx + 1)
        
        # Isolate the user's data
        user_data = interactions[interactions['user_id'] == user_id]
        
        # 80/20 Train/Test split for local evaluation
        train_df, test_df = train_test_split(user_data, test_size=0.2, random_state=42)
        
        # Generate the taste profile for the smart contract
        profile = extract_taste_profile(user_data, recipes, user_id, ingredient_map)
        
        for swarm_id in profile['swarm_ids']:
            swarm_id_counts[swarm_id] = swarm_id_counts.get(swarm_id, 0) + 1
        
        # Write to the simulated pod
        train_df.to_csv(os.path.join(client_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(client_dir, 'test.csv'), index=False)
        
        with open(os.path.join(client_dir, 'profile.json'), 'w') as f:
            json.dump(profile, f, indent=4)
            
        print(f"Client {idx + 1} provisioned. Swarm IDs: {profile['swarm_ids']}, Taste Profiles: {profile['taste_profiles']}, Interactions: {len(user_data)}")
        print(f"    Full Score Breakdown: {profile['debug_taste_scores']}\n")

    print("\n--- Swarm Assignment Summary ---")
    SWARM_ID_TO_TASTE = {v: k for k, v in TASTE_TO_SWARM_ID.items()}
    
    # Sort by swarm ID for consistent output
    sorted_counts = sorted(swarm_id_counts.items())

    for swarm_id, count in sorted_counts:
        taste_name = SWARM_ID_TO_TASTE.get(swarm_id, "Unknown")
        print(f"Swarm ID {swarm_id} ({taste_name.capitalize()}): {count} clients")
    print("---------------------------------")

if __name__ == "__main__":
    main()