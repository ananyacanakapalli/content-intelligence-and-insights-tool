# scripts/load_data_to_db.py

import pandas as pd
from sqlalchemy import create_engine, text
from load_kaggle_data import users, movies, ratings  # import the dataframes

# --- STEP 1: Connect to PostgreSQL ---
# Make sure PostgreSQL is running and you have a database called 'content_engagement'
engine = create_engine('postgresql://ananyacanakapalli:Ast9<3anan@localhost/content_engagement')


# --- STEP 2: Create tables if they don't exist ---
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INT PRIMARY KEY,
            gender CHAR(1),
            age INT,
            occupation INT,
            zip_code VARCHAR(20)
        );
    """))
    
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS shows (
            show_id INT PRIMARY KEY,
            title VARCHAR(255),
            genres VARCHAR(255)
        );
    """))
    
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS viewing_logs (
            user_id INT REFERENCES users(user_id),
            show_id INT REFERENCES shows(show_id),
            rating INT,
            timestamp BIGINT,
            PRIMARY KEY (user_id, show_id)
        );
    """))

# --- STEP 3: Load data into tables ---
# Users
users.to_sql('users', engine, if_exists='replace', index=False)

# Shows (movies)
movies.rename(columns={'movie_id': 'show_id'}, inplace=True)
movies.to_sql('shows', engine, if_exists='replace', index=False)

# Viewing logs (ratings)
ratings.rename(columns={'movie_id': 'show_id'}, inplace=True)
ratings.to_sql('viewing_logs', engine, if_exists='replace', index=False)

print("Data loaded successfully into PostgreSQL!")
