import torch
import pandas as pd
from model_training import recommendation_model
from data_loading import df
from device_setup import device

def recommend_top_movies(model, user_id, all_movies, seen_movies, device, k=5, batch_size=100):
    model.eval()
    unseen_movies = [m for m in all_movies if m not in seen_movies]
    predictions = []

    with torch.no_grad():
        for i in range(0, len(unseen_movies), batch_size):
            batch_unseen_movies = unseen_movies[i:i+batch_size]
            user_tensor = torch.tensor([user_id] * len(batch_unseen_movies)).to(device)
            movie_tensor = torch.tensor(batch_unseen_movies).to(device)
            predicted_ratings = model(user_tensor, movie_tensor).view(-1).tolist()
            predictions.extend(zip(batch_unseen_movies, predicted_ratings))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_k_movies = [movie_id for movie_id, _ in predictions[:k]]
    return top_k_movies

def get_movies_with_genres(movie_ids, df_movies):
    # Select the relevant movies and create a new DataFrame
    movies_with_genres = df_movies[df_movies['movieId'].isin(movie_ids)].copy()
    # Concatenate movie titles with their genres
    movies_with_genres['title_with_genres'] = movies_with_genres[['title', 'genres']].agg(' - '.join, axis=1)
    return movies_with_genres['title_with_genres'].tolist()

# Load movie titles and genres
df_movies = pd.read_csv("ml-latest-small/movies.csv")

# Prepare all_movies and seen_movies
all_movies = df['movieId'].unique().tolist()
user_id = 1 # A random userId
seen_movies = set(df[df['userId'] == user_id]['movieId'].tolist())

# Get recommendations
recommendations = recommend_top_movies(
    recommendation_model, user_id, all_movies, seen_movies, device
)

# Get movie titles with genres for recommended and seen movies
recommended_movies_with_genres = get_movies_with_genres(recommendations, df_movies)

# For the user's top 10 rated seen movies, get movies with genres
user_top_ten_seen_movies = df[df['userId'] == user_id].sort_values(by="rating", ascending=False).head(10)
seen_movies_with_genres = get_movies_with_genres(user_top_ten_seen_movies['movieId'], df_movies)

print(f"Recommended movies:\n\n{recommended_movies_with_genres}\n\nbased on these movies the user has watched:\n\n{seen_movies_with_genres}")
