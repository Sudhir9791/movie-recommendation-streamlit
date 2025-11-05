# 🎬 Movie Recommendation System

An interactive **Movie Recommendation Web App** built with **Streamlit** and **scikit-learn**.  
It recommends movies similar to your favorite titles using **TF-IDF vectorization** and **cosine similarity**.

---

## 🌟 Features

- 🔍 Search any movie and get top similar recommendations  
- 🧠 Uses TF-IDF and Cosine Similarity for content-based filtering  
- 🪶 Clean, responsive Streamlit UI  
- ⚡ Fast, lightweight, and ready to deploy  

---

## 🖼️ Demo

🎯 **Live App:** [https://sudhir9791-movie-recommendation-streamlit-app-2m15iv.streamlit.app/)  
*(Replace with your actual deployed link after publishing.)*

---

## 🧩 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| Language | Python |
| Framework | Streamlit |
| Machine Learning | scikit-learn (TF-IDF + Cosine Similarity) |
| Data Handling | pandas |

---

## 🧠 How It Works

1. The movie dataset (`movies.csv`) contains movie titles and their text descriptions (overview, tags, or genres).
2. The app converts each movie description into TF-IDF vectors.
3. It computes cosine similarity between movies to find the closest matches.
4. When a user searches for a movie, the app displays the most similar titles and similarity scores.

---

## ⚙️ Installation & Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/movie-recommendation-streamlit.git
cd movie-recommendation-streamlit
