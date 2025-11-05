import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches


# ---------------------------------------------------------
# 🎨 Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")
st.title("🎥 Movie Recommendation System")
st.markdown("Find movies similar to your favorites using TF-IDF and Cosine Similarity.")


# ---------------------------------------------------------
# 📂 Load Dataset (Always from local movies.csv)
# ---------------------------------------------------------
DATA_PATH = "movies.csv"

if not os.path.exists(DATA_PATH):
    st.error("⚠️ 'movies.csv' not found in this folder. Please place the file beside app.py and restart.")
    st.stop()

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# 🧠 Load & Process Data (Cached)
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(df):
    possible_cols = ["overview", "description", "tags", "genres"]
    text_col = next((col for col in possible_cols if col in df.columns), None)

    if not text_col:
        st.error("No suitable text column found. Please include 'overview', 'description', 'tags', or 'genres'.")
        st.stop()

    df = df.dropna(subset=["title", text_col])
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df[text_col].astype(str))

    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()
    return df, tfidf_matrix, indices, text_col


df, tfidf_matrix, indices, text_col = load_data(df)


# ---------------------------------------------------------
# 🔍 Helper: Find Closest Matching Title
# ---------------------------------------------------------
def find_closest_title(title, indices):
    matches = get_close_matches(title.lower(), indices.index, n=1, cutoff=0.6)
    return matches[0] if matches else None


# ---------------------------------------------------------
# 🎯 Recommendation Function
# ---------------------------------------------------------
def get_recommendations(title, df, tfidf_matrix, indices, top_n=10):
    title_match = find_closest_title(title, indices)
    if not title_match:
        st.error("Movie not found. Please check spelling or try another title.")
        return pd.DataFrame(), None

    idx = indices[title_match]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:][::-1]
    sim_indices = [i for i in sim_indices if i != idx]
    scores = sim_scores[sim_indices]

    recommendations = df.iloc[sim_indices][["title", text_col]].copy()
    recommendations["Similarity"] = scores
    return recommendations, title_match


# ---------------------------------------------------------
# 🧍 Streamlit User Interface
# ---------------------------------------------------------
st.subheader("🔎 Search for a Movie")
user_input = st.text_input("Enter a movie title:", "")

# Show some random examples
sample_titles = df["title"].sample(min(5, len(df)), random_state=42).tolist()
st.caption("💡 Example titles: " + ", ".join(sample_titles))

if st.button("🎬 Recommend"):
    if user_input.strip():
        recommendations, matched_title = get_recommendations(user_input, df, tfidf_matrix, indices)
        if matched_title and len(recommendations) > 0:
            st.success(f"Top {len(recommendations)} movies similar to **'{matched_title.title()}'**:")
            cols = st.columns(2)
            for i, (_, row) in enumerate(recommendations.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"#### 🎞️ {row['title']}")
                    st.markdown(f"**Similarity Score:** {row['Similarity']:.2f}")
                    st.write(row[text_col][:400] + "...")
                    st.markdown("---")
        else:
            st.warning("No similar movies found.")
    else:
        st.warning("Please enter a movie title first.")


# ---------------------------------------------------------
# 🪶 Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & scikit-learn")
