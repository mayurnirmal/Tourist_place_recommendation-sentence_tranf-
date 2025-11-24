import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import re
import torch

app = Flask(__name__)

# -------------------------------
# Load all required files
# -------------------------------
df = pd.read_csv("processed_places.csv")
place_embeddings = torch.tensor(np.load("place_embeddings.npy"))
model = SentenceTransformer("saved_model")   # loads your saved model folder


# -------------------------------
# Extract budget from query
# -------------------------------
def extract_budget(query):
    nums = re.findall(r'\d+', query)
    if nums:
        return int(nums[0])
    return None


# -------------------------------
# Recommendation Logic
# -------------------------------
def recommend_places(query, top_k=10):

    # Step 1: extract budget
    budget = extract_budget(query)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Step 2: compute similarity scores
    scores = util.cos_sim(query_embedding, place_embeddings)[0]

    # Step 3: get top results
    top_results = torch.topk(scores, k=top_k)

    # Step 4: check if the highest similarity is below threshold
    similarity_threshold = 0.30      # adjust based on your dataset
    max_score = float(top_results.values[0])

    if max_score < similarity_threshold:
        return []   # return empty meaning "no relevant results"

    # Step 5: build result list
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        row = df.iloc[int(idx)]

        results.append({
            "Place ID": row.get("Place ID", ""),
            "Visit Link": row.get("Visit Link", ""),
            "Place Name": row.get("Place Name", ""),
            "Image Link": row.get("Image Link", ""),
            "Description": row.get("Description", ""),
            "Package": row.get("Package", ""),
            "Rating": row.get("Rating", ""),
            "Best time": row.get("Best time", ""),
            "city": row.get("city", ""),
            "state": row.get("state", ""),
            "Predicted_Tag": row.get("Predicted_Tag", ""),
            "similarity": float(score)
        })

    return results



# -------------------------------
# Flask Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    query = ""
    top_k = 10

    if request.method == "POST":
        query = request.form["query"]
        top_k = int(request.form["top_k"])
        results = recommend_places(query, top_k)

    return render_template("index.html", results=results, query=query, top_k=top_k)


if __name__ == "__main__":
    app.run(debug=True)
