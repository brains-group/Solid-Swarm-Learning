import pickle
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")

print("Path to dataset files:", path)

with open(os.path.join(path + "/ingr_map.pkl"), "rb") as file:
    ingr_map = pickle.load(file)
ingr_map.to_csv(os.path.join(path + "/ingr_map.csv"))
