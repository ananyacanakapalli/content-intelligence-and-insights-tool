import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# ------------------------------
# Database connection
# ------------------------------
DB_USER = 'ananyacanakapalli'
DB_PASSWORD = 'Ast9<3anan'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'content_engagement'

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# ------------------------------
# Load tables from DB
# ------------------------------
@st.cache_data
def load_table(table_name):
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, engine)

users = load_table('users')
shows = load_table('shows')
viewing_logs = load_table('viewing_logs')
viewing_logs['timestamp'] = pd.to_datetime(viewing_logs['timestamp'])

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.set_page_config(page_title="Netflix Analytics Platform", layout="wide")
st.title("📊 Netflix Analytics Platform")

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("Filters")

# User Filters
age_filter = st.sidebar.slider("Filter users by age", int(users['age'].min()), int(users['age'].max()),
                               (int(users['age'].min()), int(users['age'].max())))
gender_filter = st.sidebar.multiselect("Filter by gender", users['gender'].unique())
user_filter = st.sidebar.multiselect("Filter by user_id", users['user_id'].tolist())

# Show Filters
genre_options = st.sidebar.multiselect(
    "Filter by genre",
    options=list({g for sublist in shows['genres'].str.split('|') for g in sublist})
)
show_filter = st.sidebar.multiselect("Filter by show_id", shows['show_id'].tolist())

# Date Range Filter
date_range = st.sidebar.date_input("Select date range",
                                   [viewing_logs['timestamp'].min().date(), viewing_logs['timestamp'].max().date()])

# ------------------------------
# Apply Filters
# ------------------------------
filtered_users = users[(users['age'] >= age_filter[0]) & (users['age'] <= age_filter[1])]
if gender_filter:
    filtered_users = filtered_users[filtered_users['gender'].isin(gender_filter)]
if user_filter:
    filtered_users = filtered_users[filtered_users['user_id'].isin(user_filter)]

filtered_shows = shows.copy()
if genre_options:
    filtered_shows = filtered_shows[filtered_shows['genres'].apply(lambda x: any(g in x for g in genre_options))]
if show_filter:
    filtered_shows = filtered_shows[filtered_shows['show_id'].isin(show_filter)]

filtered_logs = viewing_logs[
    (viewing_logs['timestamp'].dt.date >= date_range[0]) &
    (viewing_logs['timestamp'].dt.date <= date_range[1])
]
filtered_logs = filtered_logs[filtered_logs['user_id'].isin(filtered_users['user_id'])]
filtered_logs = filtered_logs[filtered_logs['show_id'].isin(filtered_shows['show_id'])]

# ------------------------------
# Key Metrics
# ------------------------------
st.header("📈 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", filtered_users['user_id'].nunique())
col2.metric("Total Shows", filtered_shows['show_id'].nunique())
col3.metric("Total Ratings", len(filtered_logs))

# ------------------------------
# Data Tables
# ------------------------------
st.header("👤 Users Table")
st.dataframe(filtered_users)

st.header("🎬 Shows / Movies Table")
st.dataframe(filtered_shows)

st.header("📊 Viewing Logs / Ratings Table")
st.dataframe(filtered_logs)

# ------------------------------
# Analytics Visualizations
# ------------------------------
st.header("📊 Analytics")

# Top 10 Shows by Average Rating
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Shows by Average Rating")
    top_shows = filtered_logs.groupby('show_id')['rating'].mean().sort_values(ascending=False).head(10).reset_index()
    top_shows = top_shows.merge(filtered_shows[['show_id','title']], on='show_id')
    st.bar_chart(top_shows.set_index('title')['rating'])

with col2:
    st.subheader("Ratings Distribution")
    st.bar_chart(filtered_logs['rating'].value_counts().sort_index())

# Top Users by Ratings Count
st.subheader("Top Users by Number of Ratings")
top_users_count = filtered_logs.groupby('user_id').size().sort_values(ascending=False).head(10).reset_index(name='ratings_count')
top_users_count = top_users_count.merge(filtered_users[['user_id']], on='user_id')
st.dataframe(top_users_count)

# Top Users by Average Rating
st.subheader("Top Users by Average Rating (min 5 ratings)")
user_avg = filtered_logs.groupby('user_id')['rating'].mean()
user_count = filtered_logs.groupby('user_id').size()
top_users_avg = pd.DataFrame({'avg_rating': user_avg, 'ratings_count': user_count})
top_users_avg = top_users_avg[top_users_avg['ratings_count'] >= 5].sort_values('avg_rating', ascending=False).head(10)
top_users_avg = top_users_avg.merge(filtered_users[['user_id']], left_index=True, right_on='user_id')
st.dataframe(top_users_avg)

# Genre Popularity & Average Rating
st.subheader("Top Genres by Number of Ratings")
genre_exploded = filtered_shows.assign(genres=filtered_shows['genres'].str.split('|')).explode('genres')
genre_counts = filtered_logs.merge(genre_exploded[['show_id','genres']], on='show_id').groupby('genres').size().sort_values(ascending=False)
st.bar_chart(genre_counts)

st.subheader("Average Rating per Genre")
genre_avg = filtered_logs.merge(genre_exploded[['show_id','genres']], on='show_id').groupby('genres')['rating'].mean().sort_values(ascending=False)
st.bar_chart(genre_avg)

# Trending Shows (last 30 days)
st.subheader("Trending Shows (Last 30 Days)")
recent_logs = filtered_logs[filtered_logs['timestamp'] >= (filtered_logs['timestamp'].max() - pd.Timedelta(days=30))]
trending = recent_logs.groupby('show_id').size().sort_values(ascending=False).head(10).reset_index(name='views')
trending = trending.merge(filtered_shows[['show_id','title']], on='show_id')
st.bar_chart(trending.set_index('title')['views'])

# User Activity Over Time
st.subheader("User Activity Over Time")
daily_activity = filtered_logs.groupby(filtered_logs['timestamp'].dt.date).size()
st.line_chart(daily_activity)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Developed by **Ananya Canakapalli** | Netflix Analytics Platform Demo")
