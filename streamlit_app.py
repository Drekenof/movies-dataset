# =============================
# IMPORT LIBRAIRIES
# =============================
# pip install scikit-learn
# pip install streamlit

# Pour l'embbedding de Summary
# pip install -U sentence-transformers

# streamlit run streamlit_app.py

import streamlit as st
import pandas as pd


# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data_film():
    df_films = pd.read_csv("DIM_FILM_LIST_FINAL_short_for_quick_work.csv",
    nrows = 1000,
    encoding='latin1')
    return df_films

df_films = load_data_film()


@st.cache_data
def load_data_person():
    df_person = pd.read_csv(
    'DIM_PERSON_LIST_FINAL_short_for_quick_work.csv',
    nrows = 1000,
    encoding='latin1')
    return df_person

df_person = load_data_person()

# =============================
# =============================
# =============================
# =============================
# =============================
# MACHINE LEARNING
# =============================
# =============================
# =============================
# =============================
# =============================

import pandas as pd
import io
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

# df_films.head()
# len(df_films)

# Création de DECADE pour réduire le nombre de catégories à l'inverse de YEAR
df_films["DECADE"] = (df_films["YEAR"] // 10) * 10


# =============================
# 1. FEATURES SELECTION
# =============================
# 1. Sélection des features les plus pertinentes pour définir la similarité
# print(df_films.columns)
# ['ID_FILM', 'TYPE', 'BUDGET', 'SUMMARY',
# 'POSTER_PATH', 'REVENUE', 'YEAR', 'DURATION_MINUTES', 'GENRES',
# 'IMDB_RATING', 'IMDB_VOTE_COUNT', 'TMDB_RATING', 'TMDB_VOTE_COUNT', 'BEST_RATING', 'SOURCE_TO_KEEP',
# 'TITLE_ORIGINAL', 'TITLE_FR', 'TITLE_EN',
# 'ACTOR', 'COMPOSER', 'DIRECTOR', 'DECADE']

cols_num = ["BUDGET", "REVENUE", "DURATION_MINUTES", "BEST_RATING"]
cols_cat_simple = ["TYPE", "DECADE"]
cols_cat_multi = ["GENRES", "ACTOR", "DIRECTOR", "COMPOSER"]
cols_title = ["TITLE_ORIGINAL", "TITLE_FR", "TITLE_EN"]
col_text = "SUMMARY"


# =============================
# 2. Standardisation et entrainement de toutes les données
# =============================
# Pas de pipeline de transformations
# Les transformations sont faites à chaque étape des différents types de col
# Plus facile à traiter individuellement et meilleure intégration avec MultiLabelBinarizer et TF-IDF

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, issparse
# from sentence_transformers import SentenceTransformer   # pour l'embedding de Summary


# -------------------------
# Correction des valeurs manquantes qui génèrent une erreur au ML
# -------------------------

# variable_list = [cols_num, cols_cat_simple, cols_cat_multi, cols_title, col_text]

# for variable in variable_list:
#   print(f'{variable=}')
#   print(df_films[variable].isna().sum())
#   print()


df_films['BUDGET'] = df_films['BUDGET'].fillna(0)
df_films['REVENUE'] = df_films['REVENUE'].fillna(0)
df_films['BEST_RATING'] = df_films['BEST_RATING'].fillna(0)
df_films = df_films.fillna("")


# -------------------------
# Cols numériques
# -------------------------
scaler = StandardScaler()
X_num = scaler.fit_transform(df_films[cols_num])
X_num = csr_matrix(X_num)


# -------------------------
# Cols cat simple
# -------------------------
ohe_simple = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat_simple = ohe_simple.fit_transform(df_films[cols_cat_simple])
X_cat_simple = csr_matrix(X_cat_simple)


# -------------------------
# Cols cat multi
# -------------------------
# Pour chaque colonne, spliter par | et faire une liste avec les strings
# nettoie les éléments pour éviter des éléments vides dans la liste
# au cas où il y aurait des | au début ou à la fin ou || ou | |
for col in ['GENRES', 'ACTOR', 'DIRECTOR', 'COMPOSER']:
    df_films[col + '_SPLIT'] = df_films[col].fillna('').apply(
        lambda x: [string.strip() for string in x.split("|") if string.strip() != ""]
    )

mlb_genres = MultiLabelBinarizer()
X_genres = mlb_genres.fit_transform(df_films["GENRES_SPLIT"])

mlb_actor = MultiLabelBinarizer()
X_actor = mlb_actor.fit_transform(df_films["ACTOR_SPLIT"])

mlb_director = MultiLabelBinarizer()
X_director = mlb_director.fit_transform(df_films["DIRECTOR_SPLIT"])

mlb_composer = MultiLabelBinarizer()
X_composer = mlb_composer.fit_transform(df_films["COMPOSER_SPLIT"])

# Retire les colonnes intermédiaires pour nettoyer
df_films = df_films.drop(
    columns=["GENRES_SPLIT", "ACTOR_SPLIT", "DIRECTOR_SPLIT", "COMPOSER_SPLIT"]
    )


# -------------------------
# Cols texte Summary
# -------------------------
# Version TF-IDF
# Fonctionne par comptage de mots. Repère les mots rares + Différencie les films par vocabulaire
# Pros : Simple, rapide, sparse
# Cons : Ne comprend pas le sens + les synonymes + la paraphrase. Dépend énormément de la langue et du style
# Exemple : “A retired hitman seeks revenge.” vs “An old assassin comes back for vengeance.”
#           Humain : films très proches
#           TF-IDF : presque aucun mot en commun -> similarité faible

tfidf = TfidfVectorizer(
    max_features = 500,
    stop_word = "english"
    )

X_summary = tfidf.fit_transform(df_films[col_text].fillna(""))


# Version Embeddings
# Un embedding transforme un texte en vecteur dense de nombres qui capture le sens global.
# Deux textes avec le même sens mais un vocabulaire potentiellement différent auront des vecteurs proches dans l’espace.
# Ne fonctionne pas sur google collab du fait de la limitation de RAM

# embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

# X_summary = embeddings_model.encode(
#    df_films["SUMMARY"].fillna("").tolist(),
#    show_progress_bar = True
#    )


# -------------------------
# Cols Title
# -------------------------
# Création d'une colonne title all en concaténant ORIGINAL + FR + EN
# Il faut éviter la déduplication d'un titre qui serait présent en "ORIGINAL" et "FR" ou "EN"
# Pour le vectorizer :
# pas de stopwords car les titres sont courts et chaque mot a son importance
# avec IDF, cela peut être suffisant pour traiter les stopwords car plus un mot apparaît souvent, moins il compte
# et bigramme pour garder le sens des titres

def build_title_all(row):
    titles = []

    if pd.notna(row["TITLE_ORIGINAL"]):
        titles.append(row["TITLE_ORIGINAL"].strip())

    # split if not empty
    if pd.notna(row["TITLE_FR"]):
        titles.extend(row["TITLE_FR"].split("|"))

    if pd.notna(row["TITLE_EN"]):
        titles.extend(row["TITLE_EN"].split("|"))

    # nettoyage
    titles = [
        title.strip().lower()
        for title in titles
        if title.strip() != ""
    ]

    # déduplication pour supprimer les doublons entre original / fr / en. Exemple : ['autoroute du paradis', 'paradise highway']
    titles = list(set(titles))

    # join insère " " ENTRE les éléments de la liste, pas au début ou à la fin. Exemple : "autoroute du paradis paradise highway"
    return " ".join(titles)

df_films["TITLE_ALL"] = df_films.apply(build_title_all, axis=1)

tfidf_title = TfidfVectorizer(
    max_features = 200,
    ngram_range = (1, 2),
    lowercase = True
    )

X_title = tfidf_title.fit_transform(df_films["TITLE_ALL"])


# -------------------------
# Features normalization
# -------------------------
# Besoin de faire une normalization car même avec une pondération (étape suivante), les features n'ont pas la même échelle.
# La normalisation L2 garantit que chaque film contribue par son “profil” et non par la quantité brute de features, ce qui rend la similarité cosinus fiable et les poids interprétables.
# Cela permet d'annuler la domination de Summary qui génère énormément de colonne.
X_num = normalize(X_num, norm="l2")
X_cat_simple = normalize(X_cat_simple, norm="l2")
X_genres = normalize(X_genres, norm="l2")
X_actor = normalize(X_actor, norm="l2")
X_director = normalize(X_director, norm="l2")
X_composer = normalize(X_composer, norm="l2")
X_summary = normalize(X_summary, norm="l2")
X_title = normalize(X_title, norm="l2")



# =========================
# 3. FEATURES PONDERATION
# =========================
X_num_weighted = hstack([
    X_num[:, 0] * 0.1,  # BUDGET
    X_num[:, 1] * 0.1,  # REVENUE
    X_num[:, 2] * 0.8,  # DURATION_MINUTES
    X_num[:, 3] * 0.8   # BEST_RATING
    ])

X_cat_simple_weighted = hstack([
    X_cat_simple[:, 0] * 0.1,   # TYPE
    X_cat_simple[:, 1] * 0.8    # DECADE
    ])

X_genres_weighted = X_genres * 1.0
X_actor_weighted = X_actor * 0.8
X_director_weighted = X_director * 2.0
X_composer_weighted = X_composer * 0.6
X_summary_weighted = X_summary * 0.8
X_title_weighted = X_title * 2.0


# =========================
# 4. Matrix conversion for hstack
# =========================
# scipy.sparse.hstack n’aime pas mixer dense et sparse
# X_summary est une matrice dense NumPy, alors que X_num_weighted, X_genres_weighted, etc. sont des matrices creuses
# MultiLabelBinarizer retourne du NumPy dense par défaut
# Il faut convertir les matrices en matrices creuses avec csr_matrix avant le hstack
# Faire explicitement la conversion élimine le risque de conversion invisible et facilité le debbugage
# La surcharge processeur est considérée comme négligeable

def ensure_csr(matrix):
    return matrix if issparse(matrix) and matrix.format == "csr" else csr_matrix(matrix)

matrix_list = [X_num_weighted, X_cat_simple_weighted, X_genres_weighted, X_actor_weighted, X_director_weighted, X_composer_weighted, X_summary_weighted, X_title_weighted]

matrix_list = [ensure_csr(matrix) for matrix in matrix_list]


# =========================
# 5. Concatenate all features
# =========================
# hstack concatène toutes les matrices pour créer une seule grande matrice de features, qui contient toutes les informations pour chaque film, ligne par ligne.
# hstack retourne une matrice sparse, qui peut être COO, CSC ou CSR.
# la fonction de recherche de film utilise la méthode .getrow
# Cette méthode fonctionne très bien avec des matrices csr
# csr_matrix(hstack(...)) reconstruit une matrice : moins optimisé que .tocsr()
X = hstack(matrix_list).tocsr()


# =========================
# 6. Crée une instance de NearestNeighbors sur les données standardisées.
# =========================
from sklearn.neighbors import NearestNeighbors

model_NN = NearestNeighbors(
    n_neighbors=5,
    metric="cosine"
    )


# =========================
# 7. Entraînement du modèle
# =========================
model_NN.fit(X)



# =========================
# 8. Recherche du film rentré par l'utilisateur dans notre liste de film
# =========================
def find_movie_index(input_title, df = df_films):
    input_title = input_title.lower().strip()

    # films dont le titre contient la saisie
    matches = df[df["TITLE_ALL"].str.contains(input_title, case=False, na=False)]

    if len(matches) == 0:
        return None

    # si plusieurs résultats, on prend le premier
    return matches.index[0]


# =========================
# 9. Recommandation en fonction du titre recherché
# =========================
import numpy as np

def recommend_movies(user_query, desired_number_of_recommendations):

    model = model_NN
    X_features = X
    df = df_films

    # utilisation de la fonction find_movie_index qui cherche l'input de l'utilisateur dans TITLE ALL
    input_film_index = find_movie_index(user_query, df)

    if input_film_index is None:
        raise ValueError(f"Aucun film trouvé pour le titre : '{user_query}'")

    # Récupère le vecteur du film
    movie_vector = X_features.getrow(input_film_index)

    # Recherche les voisins
    distances, indices = model.kneighbors(movie_vector, n_neighbors = desired_number_of_recommendations + 1)

    # On enlève le premier résultat (c'est le film lui-même)
    similar_indices = indices.flatten()[1:]

    # Drop colonne intermédiaire et retourne les titres des films similaires
    return df.iloc[[input_film_index]], df.drop(columns=["TITLE_ALL"]).iloc[similar_indices]












# =============================
# =============================
# =============================
# =============================
# =============================
# STREAMLIT
# =============================
# =============================
# =============================
# =============================
# =============================

# =============================
# EN-TÊTE
# =============================
st.set_page_config(page_title="CinéMad", page_icon="🎬")
st.title("🎬 CinéMad")

st.write(
    """
    Bienvenu dans les salles de cinéma de la Creuse !
    CinéMad est votre assistant pour trouver des films qui correspondent à vos envies du moment.
    """
    )

st.write(
    "Liste des titres uniques dans la base de données :"
    )
st.dataframe(
    sorted(df_films['TITLE_ALL'].unique()),
    width='stretch'
    )

st.write(
    "Liste des titres d'origine en doublon dans la base de données :"
    )
st.dataframe(
    df_films[df_films['TITLE_ORIGINAL'].duplicated()][['TITLE_ORIGINAL']],
    width='stretch'
)

# =============================
# UI
# =============================
radio_mode = st.radio(
    "Mode de recherche",
    ["Film", "Acteur", "Réalisateur", "Compositeur"],
    horizontal = True
    )

# Set placeholder based on radio selection
if radio_mode == "Film":
    input_caption = "Entrez un nom de film. Nous vous recommanderons 5 films proches de celui-ci, selon de nombreux critères."
    input_placeholder = "Inception • Batman • Harry Potter"
elif radio_mode == "Acteur":
    input_caption = "Entrez un nom d'acteur ou d'actrice. Nous afficherons la liste des films de notre liste auxquels il/elle a participé."
    input_placeholder = "Henry Cavill • Angelina Jolie • Brigitte Bardot"
elif radio_mode == "Réalisateur":
    input_caption = "Entrez un nom de réalisateur ou réalisatrice. Nous afficherons la liste des films de notre liste auxquels il/elle a participé."
    input_placeholder = "Christopher Nolan • Jean-Luc Godard • Alfred Hitchcock"
elif radio_mode == "Compositeur":
    input_caption = "Entrez un nom de compositeur ou compositrice. Nous afficherons la liste des films de notre liste auxquels il/elle a participé."
    input_placeholder = "Hans Zimmer • Ennio Morricone • John Williams"

st.caption(input_caption)

user_query = st.text_input(
    "Ta recherche",
    placeholder = input_placeholder
    )

sort_label = st.selectbox(
    "Trier les résultats",
    ["Pertinence (ML)", "Plus récents", "Mieux notés", "Plus populaires"],
    index = 0  # si on veux "Plus récents" par défaut, mets 1
    )

sort_map = {
    "Pertinence (ML)": "similar",
    "Plus récents": "recent",
    "Mieux notés": "rating",
    "Plus populaires": "votes"
    }

sort_mode = sort_map.get(sort_label, "recent")


# =============================
# ...
# =============================
desired_number_of_recommendations = 5

if st.button("Rechercher"):
    try:
        user_query = user_query.strip()
        if not user_query:
            st.warning("Entrez un texte.")
            st.stop()

        with st.spinner("Recherche en cours…"):
            if radio_mode == "Film" :
                searched_film_df, recommended_movies_df = recommend_movies(user_query, desired_number_of_recommendations)
            #     df = films_by_actor(user_query, n = desired_number_of_recommendations)
            # elif radio_mode == "Réalisateur" :
            #     df = films_by_director(user_query, n = desired_number_of_recommendations)
            # elif radio_mode == "Compositeur" :
            #     df = films_by_composer(user_query, n = desired_number_of_recommendations)

        # st.success(f"✅ {radio_mode} — {desired_number_of_recommendations} résultats")
        # render_cards(df_films, k = desired_number_of_recommendations)

    except Exception as e:
        st.error(str(e))

st.dataframe(
    searched_film_df,
    width='stretch'
    )

st.dataframe(
    recommended_movies_df,
    width='stretch'
    )

# =============================
# ...
# =============================
genre_list = (
    df_films['GENRES']
    .str.split('|')   # sépare les genres
    .explode()        # met un genre par ligne
    .str.strip()      # enlève les espaces
    .str.title()      # met en forme (Action, Drama, etc.)
    .unique()         # genres uniques
    .tolist()         # liste Python (optionnel)
)

filter_genres = st.multiselect(
    "Genres",
    genre_list
    )



if filter_genres: # si un filtre est sélectionné, afficher le df filtré 
    filter_operator = st.radio(
        "Opérateur",
        ("or", "and"),
        horizontal=True
        )

    if filter_operator == "or": # Au moins un genre sélectionné
        pattern = "|".join(filter_genres)
        mask = df_films["GENRES"].str.contains(pattern, case=False, na=False)

    elif filter_operator == "and": # Tous les genres sélectionnés doivent être listés
        mask = df_films["GENRES"].apply(
            lambda x: all(genre.lower() in x.lower() for genre in filter_genres)
        )

    df_films_to_display = df_films[mask]
else: # si aucun filtre sélectionné, afficher le df entier
    df_films_to_display = df_films

st.dataframe(
    df_films_to_display,
    width='stretch'
    )