"""
Property Matching System — Case Study 2
Author: Data Science Candidate
Description:
    Matches user preferences with property listings and calculates a
    weighted Match Score (0–100) for every user-property pair.

Methodology Overview:
    1. Numerical Score  — budget, bedrooms, bathrooms similarity via
       cosine similarity on normalized vectors.
    2. Text Score       — TF-IDF cosine similarity between user's
       qualitative description and property description.
    3. Final Score      — weighted blend:
         60% numerical + 40% text
    Justification: Numerical features provide hard constraints
    (budget, room count); qualitative descriptions capture soft
    preferences that numbers cannot encode (e.g., "ocean view",
    "eco-friendly", "mountain cabin").
"""

# ─────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────
DATA_PATH = "data.xlsx"   # update path if needed

users_raw = pd.read_excel(DATA_PATH, sheet_name="User Data")
props_raw = pd.read_excel(DATA_PATH, sheet_name="Property Data")

print("✅ Data loaded")
print(f"   Users:      {users_raw.shape}")
print(f"   Properties: {props_raw.shape}")
print()

# ─────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────
def parse_budget(val):
    """Convert '$500k', '$1.2M', etc. → float (thousands)."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").replace(",", "").strip().upper()
    if "M" in s:
        return float(s.replace("M", "")) * 1000
    if "K" in s:
        return float(s.replace("K", ""))
    return float(s) / 1000  # assume raw dollars → convert to $k

users = users_raw.copy()
props = props_raw.copy()

# --- Rename for consistency ----------------------------------------
users.columns = [c.strip() for c in users.columns]
props.columns = [c.strip() for c in props.columns]

users.rename(columns={
    "User ID": "user_id",
    "Budget": "budget",
    "Bedrooms": "bedrooms",
    "Bathrooms": "bathrooms",
    "Qualitative Description": "description",
}, inplace=True)

props.rename(columns={
    "Property ID": "property_id",
    "Price": "price",
    "Bedrooms": "bedrooms",
    "Bathrooms": "bathrooms",
    "Living Area (sq ft)": "sqft",
    "Qualitative Description": "description",
}, inplace=True)

# --- Parse money columns -------------------------------------------
users["budget"] = users["budget"].apply(parse_budget)
props["price"]  = props["price"].apply(parse_budget)

# --- Handle missing values -----------------------------------------
for df, num_cols in [
    (users, ["budget", "bedrooms", "bathrooms"]),
    (props, ["price",  "bedrooms", "bathrooms", "sqft"]),
]:
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    df["description"].fillna("", inplace=True)

# --- Normalize numerical features (0–1) ---------------------------
u_num = users[["budget", "bedrooms", "bathrooms"]].values.astype(float)
p_num = props[["price",  "bedrooms", "bathrooms"]].values.astype(float)

scaler = MinMaxScaler()
all_num = np.vstack([u_num, p_num])
scaler.fit(all_num)
u_norm = scaler.transform(u_num)
p_norm = scaler.transform(p_num)

print("✅ Preprocessing done")
print(f"   Users missing values:      {users.isnull().sum().sum()}")
print(f"   Properties missing values: {props.isnull().sum().sum()}")
print()

# ─────────────────────────────────────────────────────────────────
# 3. Match Score Calculation
# ─────────────────────────────────────────────────────────────────
# --- 3a. Numerical similarity (cosine) ----------------------------
# Shape: (n_users, n_properties)
num_sim = cosine_similarity(u_norm, p_norm)

# --- 3b. Text similarity (TF-IDF cosine) --------------------------
all_descriptions = (
    list(users["description"].values) + list(props["description"].values)
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=5000,
    min_df=1,
)
tfidf_matrix = vectorizer.fit_transform(all_descriptions)

n_users = len(users)
u_tfidf = tfidf_matrix[:n_users]
p_tfidf = tfidf_matrix[n_users:]
text_sim = cosine_similarity(u_tfidf, p_tfidf)

# --- 3c. Weighted blend -------------------------------------------
W_NUM  = 0.60   # weight for numerical similarity
W_TEXT = 0.40   # weight for text/semantic similarity

match_raw = W_NUM * num_sim + W_TEXT * text_sim        # (n_users, n_props)
match_score = np.round(match_raw * 100, 2)             # scale to 0–100

print("✅ Match scores calculated")
print(f"   Score matrix shape: {match_score.shape}")
print(f"   Score range:        {match_score.min():.1f} – {match_score.max():.1f}")
print()

# --- Build results DataFrame -------------------------------------
rows = []
for i, uid in enumerate(users["user_id"]):
    for j, pid in enumerate(props["property_id"]):
        rows.append({
            "user_id":      uid,
            "property_id":  pid,
            "num_score":    round(num_sim[i, j] * 100, 2),
            "text_score":   round(text_sim[i, j] * 100, 2),
            "match_score":  match_score[i, j],
        })

results = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────
# 4. Top-N Recommendations per User
# ─────────────────────────────────────────────────────────────────
TOP_N = 3

top_matches = (
    results
    .sort_values("match_score", ascending=False)
    .groupby("user_id")
    .head(TOP_N)
    .reset_index(drop=True)
)

print(f"✅ Top-{TOP_N} matches per user (showing first 10 rows):")
print(top_matches.head(10).to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

# --- 5a. Full Heatmap: all users × all properties -----------------
fig, ax = plt.subplots(figsize=(16, 10))
heat_data = pd.DataFrame(
    match_score,
    index=[f"U{uid}" for uid in users["user_id"]],
    columns=[f"P{pid}" for pid in props["property_id"]],
)
sns.heatmap(
    heat_data,
    cmap="YlOrRd",
    linewidths=0.3,
    annot=False,
    ax=ax,
    cbar_kws={"label": "Match Score (0–100)"},
)
ax.set_title("Property Match Score Heatmap — All Users × All Properties",
             fontsize=14, pad=12)
ax.set_xlabel("Property ID")
ax.set_ylabel("User ID")
plt.tight_layout()
plt.savefig("heatmap_full.png")
plt.close()
print("📊 Saved: heatmap_full.png")

# --- 5b. Top-3 properties per user (bar chart, first 10 users) ----
sample_users = users["user_id"].head(10).tolist()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, uid in enumerate(sample_users):
    sub = (
        results[results["user_id"] == uid]
        .nlargest(TOP_N, "match_score")
        .sort_values("match_score", ascending=True)
    )
    labels = [f"P{pid}" for pid in sub["property_id"]]
    scores = sub["match_score"].values
    colors = ["#4CAF50" if s >= 70 else "#FF9800" if s >= 50 else "#F44336"
              for s in scores]
    axes[idx].barh(labels, scores, color=colors, edgecolor="white")
    axes[idx].set_xlim(0, 100)
    axes[idx].set_title(f"User {uid}", fontsize=11)
    axes[idx].xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    for spine in ["top", "right"]:
        axes[idx].spines[spine].set_visible(False)

fig.suptitle(f"Top-{TOP_N} Property Matches per User (Users 1–10)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("top_matches_bar.png")
plt.close()
print("📊 Saved: top_matches_bar.png")

# --- 5c. Distribution of match scores ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(results["match_score"], bins=40, color="#5C6BC0", edgecolor="white",
        alpha=0.85)
ax.axvline(results["match_score"].mean(), color="crimson", linestyle="--",
           linewidth=1.8, label=f"Mean = {results['match_score'].mean():.1f}")
ax.set_xlabel("Match Score")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of All Match Scores")
ax.legend()
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.close()
print("📊 Saved: score_distribution.png")

# --- 5d. Numerical vs Text score scatter (sample 200 pairs) ------
sample = results.sample(min(200, len(results)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(sample["num_score"], sample["text_score"],
                c=sample["match_score"], cmap="plasma",
                alpha=0.75, edgecolors="none", s=50)
plt.colorbar(sc, ax=ax, label="Match Score")
ax.set_xlabel("Numerical Score (budget, bedrooms, bathrooms)")
ax.set_ylabel("Text / Semantic Score (TF-IDF)")
ax.set_title("Numerical vs Semantic Score — Sampled Pairs")
plt.tight_layout()
plt.savefig("num_vs_text_scatter.png")
plt.close()
print("📊 Saved: num_vs_text_scatter.png")

# ─────────────────────────────────────────────────────────────────
# 6. Save results to CSV
# ─────────────────────────────────────────────────────────────────
results.to_csv("all_match_scores.csv", index=False)
top_matches.to_csv("top_match_results.csv", index=False)
print()
print("💾 Saved: all_match_scores.csv")
print("💾 Saved: top_match_results.csv")

# ─────────────────────────────────────────────────────────────────
# 7. Summary Statistics
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("SUMMARY STATISTICS")
print("=" * 55)
print(f"Total user-property pairs evaluated : {len(results)}")
print(f"Average match score                 : {results['match_score'].mean():.2f}")
print(f"Highest match score                 : {results['match_score'].max():.2f}")
print(f"Lowest  match score                 : {results['match_score'].min():.2f}")
print()

best_row = results.loc[results["match_score"].idxmax()]
print(f"Best overall match  → User {int(best_row.user_id):>2d} × Property {int(best_row.property_id):>2d}  "
      f"({best_row.match_score:.1f} / 100)")

print()
print("Top match for first 5 users:")
print(results.groupby("user_id")["match_score"]
      .max()
      .head()
      .reset_index()
      .rename(columns={"match_score": "best_score"})
      .to_string(index=False))
