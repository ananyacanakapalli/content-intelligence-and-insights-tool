# scripts/load_kaggle_data.py
import pandas as pd

# Load users
users = pd.read_csv('../data/users.dat', sep='::', engine='python',
                    names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
print("Users sample:")
print(users.head())

# Load movies
movies = pd.read_csv('../data/movies.dat', sep='::', engine='python',
                     names=['movie_id', 'title', 'genres'], encoding='latin-1')
print("\nMovies sample:")
print(movies.head())

# Load ratings
ratings = pd.read_csv('../data/ratings.dat', sep='::', engine='python',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])
print("\nRatings sample:")
print(ratings.head())

# Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Split genres into list
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Check for missing values
print("Missing values in users:", users.isnull().sum())
print("Missing values in movies:", movies.isnull().sum())
print("Missing values in ratings:", ratings.isnull().sum())

