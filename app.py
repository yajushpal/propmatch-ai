"""
Property Match — Interactive UI
Run with: streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropMatch AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream:   #FAF7F2;
    --charcoal:#1C1C1E;
    --gold:    #C9A84C;
    --gold2:   #E8C97A;
    --rust:    #C0503A;
    --sage:    #6B8F6A;
    --card-bg: #FFFFFF;
    --border:  #E8E0D4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--charcoal);
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1C1C1E 60%, #2E2A24 100%);
    border-radius: 20px;
    padding: 52px 48px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.18) 0%, transparent 70%);
}
.hero-tag {
    display: inline-block;
    background: rgba(201,168,76,0.15);
    color: var(--gold2);
    border: 1px solid rgba(201,168,76,0.35);
    border-radius: 50px;
    padding: 4px 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 900;
    color: #FFFFFF;
    line-height: 1.1;
    margin: 0 0 12px;
}
.hero h1 span { color: var(--gold); }
.hero p {
    color: rgba(255,255,255,0.6);
    font-size: 16px;
    font-weight: 300;
    max-width: 540px;
    margin: 0;
    line-height: 1.7;
}

/* Stat cards */
.stat-row { display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 140px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--gold2));
    border-radius: 0 0 16px 16px;
}
.stat-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--charcoal);
    line-height: 1;
}
.stat-sub { font-size: 12px; color: #AAA; margin-top: 4px; }

/* Score badge */
.score-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 64px; height: 64px;
    border-radius: 50%;
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}
.score-high   { background: linear-gradient(135deg, #6B8F6A, #4A7A48); }
.score-mid    { background: linear-gradient(135deg, #C9A84C, #B8922A); }
.score-low    { background: linear-gradient(135deg, #C0503A, #A03020); }

/* Match card */
.match-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 14px;
    display: flex;
    align-items: flex-start;
    gap: 18px;
    transition: box-shadow 0.2s;
}
.match-card:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.08); }
.match-info { flex: 1; }
.match-title {
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    font-weight: 700;
    color: var(--charcoal);
    margin-bottom: 6px;
}
.match-desc { font-size: 13px; color: #666; line-height: 1.6; }
.match-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.pill {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    color: #555;
    letter-spacing: 0.5px;
}
.pill.gold { background: rgba(201,168,76,0.1); border-color: rgba(201,168,76,0.4); color: #A07820; }

/* Section title */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--charcoal);
    margin-bottom: 6px;
}
.section-sub { font-size: 14px; color: #888; margin-bottom: 24px; }

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, var(--gold), transparent);
    margin: 36px 0;
    border: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--charcoal) !important;
}
section[data-testid="stSidebar"] * { color: #EEE !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: var(--gold2) !important; font-weight: 600 !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model (cached) ──────────────────────────────────────────
@st.cache_data
def load_and_compute():
    def parse_budget(val):
        if pd.isna(val): return np.nan
        s = str(val).replace("$","").replace(",","").strip().upper()
        if "M" in s: return float(s.replace("M","")) * 1000
        if "K" in s: return float(s.replace("K",""))
        return float(s) / 1000

    users = pd.read_excel("data.xlsx", sheet_name="User Data")
    props = pd.read_excel("data.xlsx", sheet_name="Property Data")

    users.columns = [c.strip() for c in users.columns]
    props.columns = [c.strip() for c in props.columns]

    users.rename(columns={"User ID":"user_id","Budget":"budget","Bedrooms":"bedrooms",
                           "Bathrooms":"bathrooms","Qualitative Description":"description"}, inplace=True)
    props.rename(columns={"Property ID":"property_id","Price":"price","Bedrooms":"bedrooms",
                           "Bathrooms":"bathrooms","Living Area (sq ft)":"sqft",
                           "Qualitative Description":"description"}, inplace=True)

    users["budget"] = users["budget"].apply(parse_budget)
    props["price"]  = props["price"].apply(parse_budget)
    for df, cols in [(users,["budget","bedrooms","bathrooms"]),(props,["price","bedrooms","bathrooms","sqft"])]:
        for c in cols: df[c].fillna(df[c].median(), inplace=True)
        df["description"].fillna("", inplace=True)

    u_num = users[["budget","bedrooms","bathrooms"]].values.astype(float)
    p_num = props[["price","bedrooms","bathrooms"]].values.astype(float)
    scaler = MinMaxScaler()
    scaler.fit(np.vstack([u_num, p_num]))
    u_norm, p_norm = scaler.transform(u_num), scaler.transform(p_num)
    num_sim = cosine_similarity(u_norm, p_norm)

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    mat = tfidf.fit_transform(list(users["description"]) + list(props["description"]))
    n = len(users)
    text_sim = cosine_similarity(mat[:n], mat[n:])

    match_score = np.round((0.6 * num_sim + 0.4 * text_sim) * 100, 2)

    rows = []
    for i, uid in enumerate(users["user_id"]):
        for j, pid in enumerate(props["property_id"]):
            rows.append({"user_id":uid,"property_id":pid,
                         "num_score":round(num_sim[i,j]*100,2),
                         "text_score":round(text_sim[i,j]*100,2),
                         "match_score":match_score[i,j]})
    results = pd.DataFrame(rows)
    return users, props, results, match_score

users, props, results, match_matrix = load_and_compute()


# ── Helper ─────────────────────────────────────────────────────────
def score_class(s):
    if s >= 58: return "score-high"
    if s >= 52: return "score-mid"
    return "score-low"

def budget_label(v):
    if v >= 1000: return f"${v/1000:.1f}M"
    return f"${int(v)}k"


# ── Hero ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">✦ AI-Powered Real Estate</div>
  <h1>Prop<span>Match</span> AI</h1>
  <p>Intelligent property matching that goes beyond filters — understanding what people
     truly want through semantic understanding and smart scoring.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ──────────────────────────────────────────────────────────
best = results.loc[results["match_score"].idxmax()]
st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-label">Total Users</div>
    <div class="stat-value">{len(users)}</div>
    <div class="stat-sub">preference profiles</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Properties</div>
    <div class="stat-value">{len(props)}</div>
    <div class="stat-sub">listings evaluated</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Pairs Scored</div>
    <div class="stat-value">{len(results)}</div>
    <div class="stat-sub">match calculations</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Score</div>
    <div class="stat-value">{results['match_score'].mean():.1f}</div>
    <div class="stat-sub">out of 100</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Best Match</div>
    <div class="stat-value">{best['match_score']:.1f}</div>
    <div class="stat-sub">User {int(best.user_id)} × Prop {int(best.property_id)}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍  Match Explorer", "📊  Analytics", "🏠  Property Browser", "⚙️  Methodology"])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — Match Explorer
# ════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-title">Find Matches</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Select a user to see their best-fit properties</div>', unsafe_allow_html=True)

        uid = st.selectbox("Select User", users["user_id"].tolist(),
                           format_func=lambda x: f"User {x}")

        user_row = users[users["user_id"] == uid].iloc[0]

        st.markdown(f"""
        <div class="match-card" style="flex-direction:column; background:#1C1C1E; border-color:#333;">
          <div style="color:var(--gold2);font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">User {uid} Profile</div>
          <div style="color:#EEE;font-size:13px;line-height:1.7;">{user_row['description'][:220]}...</div>
          <div class="match-pills" style="margin-top:12px;">
            <span class="pill gold">Budget: {budget_label(user_row['budget'])}</span>
            <span class="pill gold">{int(user_row['bedrooms'])} bed</span>
            <span class="pill gold">{int(user_row['bathrooms'])} bath</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        top_n = st.slider("Show top N matches", 3, 10, 5)
        w_num  = st.slider("Numerical weight (%)", 10, 90, 60, step=5)
        w_text = 100 - w_num
        st.caption(f"Text weight: {w_text}%")

    with col_right:
        st.markdown('<div class="section-title">Top Matches</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Best properties for User {uid} — ranked by match score</div>', unsafe_allow_html=True)

        # Recompute with custom weights
        u_idx = users[users["user_id"] == uid].index[0]
        u_num_vec = users.loc[[u_idx], ["budget","bedrooms","bathrooms"]].values.astype(float)
        p_num_mat = props[["price","bedrooms","bathrooms"]].values.astype(float)
        from sklearn.preprocessing import MinMaxScaler as MMS
        sc2 = MMS(); sc2.fit(np.vstack([u_num_vec, p_num_mat]))
        u_n2 = sc2.transform(u_num_vec); p_n2 = sc2.transform(p_num_mat)
        ns = cosine_similarity(u_n2, p_n2)[0]

        tfidf2 = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        all_d = [user_row["description"]] + list(props["description"])
        m2 = tfidf2.fit_transform(all_d)
        ts = cosine_similarity(m2[0:1], m2[1:])[0]

        scores = ((w_num/100) * ns + (w_text/100) * ts) * 100
        top_idx = np.argsort(scores)[::-1][:top_n]

        for rank, pidx in enumerate(top_idx):
            prop = props.iloc[pidx]
            sc   = round(scores[pidx], 1)
            cls  = score_class(sc)
            st.markdown(f"""
            <div class="match-card">
              <div class="score-badge {cls}">
                {sc}
              </div>
              <div class="match-info">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                  <div class="match-title">Property {int(prop.property_id)}</div>
                  <span style="font-size:11px;color:#AAA;font-weight:500;">#{rank+1} match</span>
                </div>
                <div class="match-desc">{prop['description'][:200]}...</div>
                <div class="match-pills">
                  <span class="pill gold">{budget_label(prop['price'])}</span>
                  <span class="pill">{int(prop['bedrooms'])} bed · {int(prop['bathrooms'])} bath</span>
                  <span class="pill">{int(prop['sqft']):,} sq ft</span>
                  <span class="pill">Numerical: {round(ns[pidx]*100,1)}</span>
                  <span class="pill">Semantic: {round(ts[pidx]*100,1)}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Score Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">All 25 users × 30 properties</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#FAF7F2")
        ax.set_facecolor("#FAF7F2")
        heat = pd.DataFrame(match_matrix,
                            index=[f"U{u}" for u in users["user_id"]],
                            columns=[f"P{p}" for p in props["property_id"]])
        sns.heatmap(heat, cmap="YlOrRd", linewidths=0.2, annot=False,
                    ax=ax, cbar_kws={"label":"Match Score"})
        ax.set_title("Match Score Heatmap", fontsize=13, pad=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Spread across all 750 pairs</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        fig2.patch.set_facecolor("#FAF7F2")
        ax2.set_facecolor("#FAF7F2")
        ax2.hist(results["match_score"], bins=40, color="#C9A84C", edgecolor="white", alpha=0.9)
        ax2.axvline(results["match_score"].mean(), color="#C0503A", linestyle="--",
                    linewidth=2, label=f"Mean = {results['match_score'].mean():.1f}")
        ax2.set_xlabel("Match Score", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title("Distribution of Match Scores", fontsize=13, fontweight="bold")
        ax2.legend()
        ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top Match per User</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Best scoring property for each of the 25 users</div>', unsafe_allow_html=True)
    top_per_user = results.groupby("user_id")["match_score"].max().reset_index()
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    fig3.patch.set_facecolor("#FAF7F2")
    ax3.set_facecolor("#FAF7F2")
    colors = ["#6B8F6A" if s >= 58 else "#C9A84C" if s >= 52 else "#C0503A"
              for s in top_per_user["match_score"]]
    bars = ax3.bar([f"U{u}" for u in top_per_user["user_id"]],
                   top_per_user["match_score"], color=colors, edgecolor="white", width=0.65)
    ax3.axhline(top_per_user["match_score"].mean(), color="#555", linestyle="--",
                linewidth=1.2, label="Average")
    ax3.set_ylabel("Best Match Score", fontsize=11)
    ax3.set_title("Best Property Match Score per User", fontsize=13, fontweight="bold")
    ax3.set_ylim(0, 75)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.legend()
    for bar, val in zip(bars, top_per_user["match_score"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


# ════════════════════════════════════════════════════════════════════
# TAB 3 — Property Browser
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Property Listings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Browse all 30 properties in the dataset</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        price_range = st.select_slider("Price Range", options=["$200k","$300k","$400k","$500k","$600k","$700k","$800k+"],
                                        value=("$200k","$800k+"))
    with col_f2:
        bed_filter = st.multiselect("Bedrooms", sorted(props["bedrooms"].unique().astype(int).tolist()),
                                     default=sorted(props["bedrooms"].unique().astype(int).tolist()))
    with col_f3:
        sort_by = st.selectbox("Sort by", ["Property ID","Price (Low→High)","Price (High→Low)","Bedrooms"])

    price_map = {"$200k":200,"$300k":300,"$400k":400,"$500k":500,"$600k":600,"$700k":700,"$800k+":900}
    p_min = price_map.get(price_range[0], 0)
    p_max = price_map.get(price_range[1], 9999)

    filtered = props[(props["price"] >= p_min) & (props["price"] <= p_max) &
                     (props["bedrooms"].astype(int).isin(bed_filter))]
    if sort_by == "Price (Low→High)": filtered = filtered.sort_values("price")
    elif sort_by == "Price (High→Low)": filtered = filtered.sort_values("price", ascending=False)
    elif sort_by == "Bedrooms": filtered = filtered.sort_values("bedrooms", ascending=False)

    st.caption(f"Showing {len(filtered)} of {len(props)} properties")

    cols = st.columns(3)
    for i, (_, prop) in enumerate(filtered.iterrows()):
        avg_score = results[results["property_id"] == prop["property_id"]]["match_score"].mean()
        with cols[i % 3]:
            st.markdown(f"""
            <div class="match-card" style="flex-direction:column; min-height:180px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="match-title">Property {int(prop.property_id)}</div>
                <div class="score-badge {score_class(avg_score)}" style="width:44px;height:44px;font-size:13px;">
                  {avg_score:.0f}
                </div>
              </div>
              <div class="match-desc" style="margin-top:6px;">{prop['description'][:160]}...</div>
              <div class="match-pills">
                <span class="pill gold">{budget_label(prop['price'])}</span>
                <span class="pill">{int(prop['bedrooms'])}bed·{int(prop['bathrooms'])}bath</span>
                <span class="pill">{int(prop['sqft']):,} sqft</span>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 4 — Methodology
# ════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-sub">A hybrid scoring model combining structured and semantic signals</div>

        <div class="match-card" style="flex-direction:column; gap:6px;">
          <div style="font-family:'Playfair Display',serif;font-size:16px;font-weight:700;color:var(--gold);">
            Step 1 — Data Preprocessing
          </div>
          <div class="match-desc">Budget strings ($500k, $1.2M) are parsed to numerical values. All numerical features
          (budget/price, bedrooms, bathrooms) are normalized using MinMax scaling across the combined
          user+property space so comparisons are fair.</div>
        </div>

        <div class="match-card" style="flex-direction:column; gap:6px;">
          <div style="font-family:'Playfair Display',serif;font-size:16px;font-weight:700;color:var(--gold);">
            Step 2 — Numerical Similarity (60%)
          </div>
          <div class="match-desc">Cosine similarity is computed between the normalized user preference vector
          [budget, bedrooms, bathrooms] and each property vector [price, bedrooms, bathrooms].
          This captures hard structural alignment.</div>
        </div>

        <div class="match-card" style="flex-direction:column; gap:6px;">
          <div style="font-family:'Playfair Display',serif;font-size:16px;font-weight:700;color:var(--gold);">
            Step 3 — Semantic Similarity (40%)
          </div>
          <div class="match-desc">TF-IDF vectorization (bigrams, 5k features) encodes the qualitative descriptions
          into sparse document vectors. Cosine similarity captures shared vocabulary — words like
          "fireplace", "ocean view", "eco-friendly" — that reveal genuine preference alignment.</div>
        </div>

        <div class="match-card" style="flex-direction:column; gap:6px;">
          <div style="font-family:'Playfair Display',serif;font-size:16px;font-weight:700;color:var(--gold);">
            Step 4 — Weighted Blend → Match Score
          </div>
          <div class="match-desc">Final score = (0.60 × numerical_sim + 0.40 × text_sim) × 100.
          Scores range 0–100. Weights are tunable via the Match Explorer tab.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Score Legend</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:24px;">
          <div class="match-card" style="flex-direction:column;background:#F0F7F0;border-color:#6B8F6A;">
            <div style="font-weight:700;color:#4A7A48;font-size:15px;">🟢 58 – 100 — Strong Match</div>
            <div class="match-desc">High alignment across budget, room count, and lifestyle preferences.</div>
          </div>
          <div class="match-card" style="flex-direction:column;background:#FDF8EC;border-color:#C9A84C;">
            <div style="font-weight:700;color:#A07820;font-size:15px;">🟡 52 – 57 — Good Match</div>
            <div class="match-desc">Solid numerical fit; some qualitative overlap. Worth exploring.</div>
          </div>
          <div class="match-card" style="flex-direction:column;background:#FDF0EE;border-color:#C0503A;">
            <div style="font-weight:700;color:#A03020;font-size:15px;">🔴 0 – 51 — Weak Match</div>
            <div class="match-desc">Significant mismatch in budget, size, or lifestyle expectations.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="font-size:18px;">Tech Stack</div>', unsafe_allow_html=True)
        for lib, desc in [("pandas / numpy","Data loading & preprocessing"),
                           ("scikit-learn","TF-IDF, cosine similarity, MinMax scaling"),
                           ("matplotlib / seaborn","Analytics visualizations"),
                           ("streamlit","This interactive UI")]:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--border);">
              <span style="font-weight:600;color:var(--charcoal);min-width:160px;">{lib}</span>
              <span style="font-size:13px;color:#888;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
