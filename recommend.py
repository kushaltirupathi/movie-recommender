import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean the dataset
df = pd.read_csv("movies.csv")

# Check for the correct column names
if 'movie_title' not in df.columns or 'genres' not in df.columns:
    print("❌ Error: CSV must contain 'movie_title' and 'genres' columns.")
    print(f"Found columns: {df.columns.tolist()}")
    exit()

# Use movie title and plot keywords (or genres)
df = df[['movie_title', 'genres']].dropna()
df['movie_title'] = df['movie_title'].str.strip()

# TF-IDF on genres
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['movie_title'])

def recommend(title, n=5):
    title = title.strip()
    if title not in indices:
        print(f"'{title}' not found.")
        return
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    print("\nRecommended movies:")
    print(df['movie_title'].iloc[movie_indices].to_string(index=False))

# Run the system
if __name__ == "__main__":
    user_input = input("Enter a movie title: ")
    recommend(user_input)

