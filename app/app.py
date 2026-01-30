import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# ------------------------------
# PAGE CONFIG & UI (CSS)
# ------------------------------
st.set_page_config(page_title="Content Intelligence & Insights Tool", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fffcf9 0%, #f7fff7 100%);
        color: #4f4f4f;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid #ffe5ec;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: none !important; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 15px 15px 0px 0px;
        padding: 10px 25px;
        color: #999;
        border: 1px solid #eee;
    }
    [data-baseweb="tab-highlight"] { display: none !important; }
    
    .stTabs [data-baseweb="tab"]:nth-of-type(1)[aria-selected="true"] {
        background-color: #ffe5ec !important; 
        color: #4f4f4f !important;
        border-bottom: 4px solid #ffb3c1 !important;
    }
    .stTabs [data-baseweb="tab"]:nth-of-type(2)[aria-selected="true"] {
        background-color: #f3e5f5 !important;
        color: #4f4f4f !important;
        border-bottom: 4px solid #b79ced !important;
    }
    .stTabs [data-baseweb="tab"]:nth-of-type(3)[aria-selected="true"] {
        background-color: #e3f2fd !important;
        color: #4f4f4f !important;
        border-bottom: 4px solid #a2d2ff !important;
    }

    .kpi-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        border: 1px solid #f0f0f0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
    }
    .kpi-value { font-size: 26px; font-weight: bold; }
    .kpi-label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .color-rose { color: #ff8fa3; }
    .color-lavender { color: #b79ced; }
    .color-blue { color: #a2d2ff; }
    .color-mint { color: #94d2bd; }

    .insight-card {
        background-color: #f3e5f5; color: #6a1b9a; padding: 15px; border-radius: 12px; margin-bottom: 10px; border: 1.5px solid #e1bee7; font-size: 14px; line-height: 1.4;
    }
    .insight-card-alt {
        background-color: #e8f5e9; color: #2e7d32; padding: 15px; border-radius: 12px; margin-bottom: 10px; border: 1.5px solid #c8e6c9; font-size: 14px; line-height: 1.4;
    }
    .reco-info-card {
        background-color: #e3f2fd; color: #1565c0; padding: 15px; border-radius: 12px; margin-bottom: 10px; border: 1.5px solid #bbdefb; font-size: 14px; line-height: 1.4;
    }

    .reco-card {
        background: rgba(255, 255, 255, 0.9); border-radius: 18px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }
    .reco-rose { border: 2.5px solid #ffcad4; color: #ff8fa3; }
    .reco-lavender { border: 2.5px solid #e0c3fc; color: #b79ced; }
    .reco-blue { border: 2.5px solid #bde0fe; color: #a2d2ff; }
    .reco-mint { border: 2.5px solid #c1fba4; color: #72efdd; }
    .reco-apricot { border: 2.5px solid #f9dcc4; color: #f08080; }
    .reco-lemon { border: 2.5px solid #fcf6bd; color: #e9c46a; }
    .reco-title { font-size: 19px; font-weight: bold; margin-bottom: 8px; }
    .reco-tag { font-size: 13px; color: #000 !important; margin-bottom: 10px; }
    .reco-reason { font-size: 12px; font-weight: 500; color: #000 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# DATA UTILITIES
# ------------------------------
def clean_genre_logic(x):
    cleaned = str(x).replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace("|", ", ")
    return ", ".join([g.strip() for g in cleaned.split(",") if g.strip()])

def normalize_movie_titles(df):
    def reformat(title):
        if pd.isna(title): return title
        t = str(title).strip()
        t = re.sub(r'\s*\(\d{4}\)$', '', t)
        pattern = r'^(.*),\s*(The|A|An)$'
        match = re.match(pattern, t, flags=re.IGNORECASE)
        if match:
            return f"{match.group(2)} {match.group(1).strip()}"
        return t
    df['title'] = df['title'].apply(reformat)
    return df

# ------------------------------
# DB CONNECTION
# ------------------------------
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

@st.cache_data
def load_db_data():
    u = pd.read_sql("SELECT * FROM users", engine)
    s = pd.read_sql("SELECT * FROM shows", engine)
    v = pd.read_sql("SELECT * FROM viewing_logs", engine)
    s["genres"] = s["genres"].apply(clean_genre_logic)
    s = normalize_movie_titles(s)
    v["timestamp"] = pd.to_datetime(v["timestamp"])
    return u, s, v

# ------------------------------
# SIDEBAR DATA SELECTION
# ------------------------------
st.sidebar.markdown("<h2 style='color: #ff8fa3;'>Data Control</h2>", unsafe_allow_html=True)
data_source = st.sidebar.radio("Select Dataset", ["Demo Dataset", "Custom Dataset"])

if data_source == "Custom Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        raw_upload = pd.read_csv(uploaded_file)
        st.sidebar.subheader("Map Your Columns")
        col_id = st.sidebar.selectbox("Show ID Column", raw_upload.columns)
        col_title = st.sidebar.selectbox("Title Column", raw_upload.columns)
        col_genres = st.sidebar.selectbox("Genres Column", raw_upload.columns)
        shows = raw_upload[[col_id, col_title, col_genres]].copy()
        shows.columns = ['show_id', 'title', 'genres']
        shows["genres"] = shows["genres"].apply(clean_genre_logic)
        shows = normalize_movie_titles(shows)
        users, _, viewing_logs = load_db_data()
        st.sidebar.success("Custom Data Active")
    else:
        st.sidebar.info("Awaiting CSV upload...")
        st.stop()
else:
    try:
        users, shows, viewing_logs = load_db_data()
        st.sidebar.success("Demo Data Connected")
    except Exception as e:
        st.error(f"DB Error: {e}")
        st.stop()

# ------------------------------
# CORE UI
# ------------------------------
st.title("Content Intelligence & Insights Tool")
tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Content Catalog", "Viewer Recommendations"])

# --- TAB 1: PERFORMANCE METRICS ---
with tab1:
    valid_logs = viewing_logs[viewing_logs.show_id.isin(shows.show_id)]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''<div class="kpi-card"><div class="kpi-value color-rose">{len(shows)}</div><div class="kpi-label color-rose">Total Titles</div></div>''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''<div class="kpi-card"><div class="kpi-value">{len(valid_logs)}</div><div class="kpi-label">Engagements</div></div>''', unsafe_allow_html=True)
    with col3:
        avg_r = valid_logs.rating.mean() if not valid_logs.empty else 0
        st.markdown(f'''<div class="kpi-card"><div class="kpi-value color-blue">{avg_r:.2f} ★</div><div class="kpi-label color-blue">Avg. Rating</div></div>''', unsafe_allow_html=True)
    with col4:
        top_genre = shows['genres'].str.split(', ').explode().mode()[0] if not shows.empty else "N/A"
        st.markdown(f'''<div class="kpi-card"><div class="kpi-value color-mint">{top_genre}</div><div class="kpi-label color-mint">Top Genre</div></div>''', unsafe_allow_html=True)

    st.write("---")
    st.subheader("Market Opportunity & Inventory Gap Analysis")
    
    if not valid_logs.empty:
        lib_share = shows['genres'].str.split(', ').explode().value_counts(normalize=True).reset_index()
        lib_share.columns = ['Genre', 'Library Share']
        view_share = valid_logs.merge(shows, on='show_id')['genres'].str.split(', ').explode().value_counts(normalize=True).reset_index()
        view_share.columns = ['Genre', 'Audience Engagement']
        gap_df = pd.merge(lib_share, view_share, on='Genre').fillna(0)
        gap_df['Gap'] = gap_df['Audience Engagement'] - gap_df['Library Share']
        
        chart_col, info_col = st.columns([2, 1], vertical_alignment="top")
        with chart_col:
            fig = px.bar(gap_df, x='Genre', y=['Library Share', 'Audience Engagement'], 
                         barmode='group', title="Catalog Mix vs. Audience Interest",
                         color_discrete_sequence=['#ffc8dd', '#bde0fe'],
                         labels={"value": "Market Share", "variable": "Metric Type"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        with info_col:
            st.markdown('<p style="color: black; font-weight: bold; font-size: 16px; margin-bottom: 10px;">💡 Analyst Insights</p>', unsafe_allow_html=True)
            ops = gap_df.sort_values(by='Gap', ascending=False).head(2)
            for i, (index, row) in enumerate(ops.iterrows()):
                card_class = "insight-card" if i == 0 else "insight-card-alt"
                st.markdown(f"""<div class="{card_class}"><b>{row['Genre']}</b> is underserved. Audience demand is <b>{row['Audience Engagement']*100:.1f}%</b>, but it only represents <b>{row['Library Share']*100:.1f}%</b> of our library.</div>""", unsafe_allow_html=True)
            
            st.markdown('<p style="color: black; font-weight: bold; font-size: 16px; margin-top: 20px;">🎯 Strategic Actions</p>', unsafe_allow_html=True)
            
            # --- DYNAMIC STRATEGIC ACTIONS FIX ---
            donor_genre = gap_df.sort_values(by='Gap', ascending=True).iloc[0]['Genre']
            dynamic_shift = min(round(ops.iloc[0]['Gap'] * 100), 25)
            
            if dynamic_shift > 0:
                st.markdown(f"""<div class="reco-info-card">Budget Allocation: Shift <b>{dynamic_shift}%</b> of <b>{donor_genre}</b> budget toward {ops.iloc[0]['Genre']} and {ops.iloc[1]['Genre']} to maximize ROI.</div>""", unsafe_allow_html=True)
            else:
                st.success("Catalog Alignment: Your current library mix closely matches audience demand.")
    else:
        st.warning("No viewing history found for this dataset.")

# --- TAB 2: CONTENT CATALOG ---
with tab2:
    st.subheader("Content Catalog")
    st.dataframe(shows, use_container_width=True, hide_index=True)

# --- TAB 3: VIEWER RECOMMENDATIONS ---
with tab3:
    st.subheader("Personalized Discovery by User Profile")
    target_user = st.selectbox("Select Profile ID", users.user_id.tolist())
    mlb = MultiLabelBinarizer()
    g_matrix = mlb.fit_transform(shows.genres.str.split(", "))
    s_sim_mat = pd.DataFrame(cosine_similarity(g_matrix), index=shows.show_id, columns=shows.show_id)

    def generate_recs(uid, n=6):
        logs = viewing_logs[viewing_logs.user_id == uid]
        current_catalog_ids = shows.show_id.tolist()
        valid_user_logs = logs[logs.show_id.isin(current_catalog_ids)]
        watched = valid_user_logs.show_id.tolist()
        fav_ids = valid_user_logs[valid_user_logs.rating >= 4].show_id.tolist()
        t_scores = s_sim_mat[fav_ids].sum(axis=1) if fav_ids else pd.Series(0, index=current_catalog_ids)
        t_scores = t_scores.drop(watched, errors="ignore")
        if t_scores.max() > 0: t_scores /= t_scores.max()
        final = t_scores.sort_values(ascending=False).head(n)
        res = shows[shows.show_id.isin(final.index)].copy()
        user_top_genres = shows[shows.show_id.isin(fav_ids)].genres.str.split(", ").explode().mode()
        top_genre = user_top_genres.iloc[0] if not user_top_genres.empty else ""
        res['reason'] = [f"Matches your {top_genre} history" if top_genre and top_genre in r else "Aligned with viewing patterns" for r in res['genres']]
        return res

    recs = generate_recs(target_user)
    if not recs.empty:
        pastel_classes = ["reco-rose", "reco-lavender", "reco-blue", "reco-mint", "reco-apricot", "reco-lemon"]
        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            color_class = pastel_classes[i % len(pastel_classes)]
            with cols[i % 3]:
                st.markdown(f"""
                <div class="reco-card {color_class}">
                    <div class="reco-title">{row['title']}</div>
                    <div class="reco-tag">Genres: {row['genres']}</div>
                    <div class="reco-reason">✨ {row['reason']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Personalized recommendations require matching view history.")

st.markdown("---")
st.caption("Content Intelligence & Insights Tool | Created by Ananya Canakapalli")
