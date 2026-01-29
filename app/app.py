import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Netflix Analytics Platform",
    layout="wide"
)

# ------------------------------
# UI (CSS)
# ------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #141414; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { background-color: #141414; }
    .stTabs [data-baseweb="tab"] { color: #b3b3b3; }
    .stTabs [aria-selected="true"] { color: #e50914 !important; border-bottom-color: #e50914 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# DATABASE CONNECTION
# ------------------------------
# os.getenv looks for these keys in your .env file
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ------------------------------
# DATA LOADING & CLEANING
# ------------------------------
@st.cache_data
def load_all_data():
    u = pd.read_sql("SELECT * FROM users", engine)
    s = pd.read_sql("SELECT * FROM shows", engine)
    v = pd.read_sql("SELECT * FROM viewing_logs", engine)
    
    def clean_genre_string(x):
        if isinstance(x, (list, set)):
            return ", ".join(sorted(x))
        cleaned = str(x).replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace("|", ", ")
        return ", ".join([g.strip() for g in cleaned.split(",") if g.strip()])

    s["genres"] = s["genres"].apply(clean_genre_string)
    v["timestamp"] = pd.to_datetime(v["timestamp"])
    return u, s, v

if all([DB_USER, DB_PASSWORD, DB_HOST]):
    users, shows, viewing_logs = load_all_data()
else:
    st.error("Database credentials missing. Please check your .env file.")
    st.stop()

# ------------------------------
# SIDEBAR FILTERS
# ------------------------------
st.title("Analytics Platform")

with st.sidebar:
    st.header("Global Filters")
    age_range = st.slider(
        "Age Range",
        int(users.age.min()), int(users.age.max()),
        (int(users.age.min()), int(users.age.max()))
    )
    available_genders = users.gender.unique().tolist()
    selected_genders = st.multiselect("Gender", available_genders, default=available_genders)

filtered_users = users[
    (users.age.between(age_range[0], age_range[1])) & 
    (users.gender.isin(selected_genders))
]

# ------------------------------
# APPLICATION TABS
# ------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Engagement Dashboard", "🎬 Content Catalog", "🎯 Personalized Recommendations"])

with tab1:
    st.header("User Engagement Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Shows (Avg Rating)")
        stats = (viewing_logs.groupby("show_id")["rating"].mean().sort_values(ascending=False).head(10)
                 .reset_index().merge(shows[["show_id", "title"]], on="show_id"))
        st.bar_chart(stats.set_index("title")["rating"])
    with col2:
        st.subheader("Platform Ratings Distribution")
        st.bar_chart(viewing_logs.rating.value_counts().sort_index())
    st.dataframe(filtered_users, use_container_width=True)

with tab2:
    st.header("Content Library")
    st.dataframe(shows, use_container_width=True)

with tab3:
    st.header("Personalized for You")
    target_user = st.selectbox("Select User Profile", users.user_id.tolist())
    
    # --- RECOMMENDATION SYSTEM ---
    ratings_matrix = viewing_logs.pivot_table(index="user_id", columns="show_id", values="rating").fillna(0)
    u_sim_mat = pd.DataFrame(cosine_similarity(ratings_matrix), index=ratings_matrix.index, columns=ratings_matrix.index)
    mlb = MultiLabelBinarizer()
    g_matrix = mlb.fit_transform(shows.genres.str.split(", "))
    s_sim_mat = pd.DataFrame(cosine_similarity(g_matrix), index=shows.show_id, columns=shows.show_id)

    def generate_recs(uid, n=6):
        logs = viewing_logs[viewing_logs.user_id == uid]
        watched = logs.show_id.tolist()
        u_sims = u_sim_mat[uid]
        c_scores = ratings_matrix.T.dot(u_sims) / (u_sims.sum() + 1e-9)
        favs = logs[logs.rating >= 4].show_id.tolist()
        t_scores = s_sim_mat[favs].sum(axis=1) if favs else pd.Series(0, index=shows.show_id)
        
        c_scores = c_scores.drop(watched, errors="ignore")
        t_scores = t_scores.drop(watched, errors="ignore")
        
        if c_scores.max() > 0: c_scores /= c_scores.max()
        if t_scores.max() > 0: t_scores /= t_scores.max()
        
        alpha = 0.7 if len(logs) > 5 else 0.3
        final = (alpha * c_scores + (1-alpha) * t_scores).sort_values(ascending=False).head(n)
        
        res = shows[shows.show_id.isin(final.index)].copy()
        res['score'] = res['show_id'].map(final)
        res = res.sort_values('score', ascending=False)
        
        fav_genres = shows[shows.show_id.isin(favs)].genres.str.split(", ").explode().mode().tolist()
        reasons = []
        for _, r in res.iterrows():
            overlap = set(r['genres'].split(", ")).intersection(set(fav_genres))
            reasons.append(f"Because you enjoy {list(overlap)[0]}" if overlap else "Recommended for you")
        res['reason'] = reasons
        return res

    recommendations = generate_recs(target_user)

    # --- DARK THEME ---
    cards_html = ""
    for _, row in recommendations.iterrows():
        cards_html += f"""
        <div style="min-width: 250px; background-color: #1f1f1f; border-radius: 10px; padding: 20px; border: 1px solid #333; flex-shrink: 0; color: white;">
            <div style="font-size: 18px; font-weight: 700; margin-bottom: 5px;">{row['title']}</div>
            <div style="font-size: 13px; color: #b3b3b3; text-transform: uppercase; margin-bottom: 15px;">{row['genres']}</div>
            <div style="font-size: 13px; color: #e50914; font-weight: 600; border-top: 1px solid #333; padding-top: 10px;">★ {row['reason']}</div>
        </div>
        """

    full_component_html = f"""
    <body style="background-color: #141414; margin: 0; padding: 0;">
        <div style="display: flex; gap: 20px; overflow-x: auto; padding: 15px; background-color: #141414; font-family: 'Segoe UI', sans-serif;">
            {cards_html}
        </div>
    </body>
    """
    components.html(full_component_html, height=230, scrolling=False)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Netflix Analytics Platform | Built by Ananya Canakapalli")
