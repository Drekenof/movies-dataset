# =============================
# IMPORT LIBRAIRIES
# =============================
# pip install scikit-learn
# pip install streamlit
# pip install streamlit-searchbox

# Pour l'embbedding de Summary
# pip install -U sentence-transformers

# streamlit run streamlit_app.py

import pandas as pd
import streamlit as st


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
    nrows = 4000,
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
    stop_words = "english"
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

    # films dont l'ID est contenu dans la fonction searchbox
    matches = df[df["ID_FILM"].str.contains(input_title, case=False, na=False)]

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
# DF SETUP
# =============================
# Crée une table de mapping ID_PERSON -> PERSON_NAME pour retrouver les noms des personnes
# Utilisée dans la fonction ids_to_names
mapping_table_id_to_name = dict(zip(df_person["ID_PERSON"], df_person["PERSON_NAME"]))



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

# =============================
# FONCTIONS
# =============================
def genre_split(genres_string):
#   Retourne la liste des genres uniques à partir d'une série de genres séparés par | (pipe)
    return [genre.strip().title()
            for genre in genres_string.split("|")
            if genre.strip()
            ]



def shorten(text, max_len = 240):
#   Nettoie le texte et le tronque à max_len sans couper les mots    
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + "…"


# Créer une barre de recherche avec autocomplete pour les titres de films
# Nécessite l'installation de streamlit-searchbox : pip install streamlit-searchbox
from streamlit_searchbox import st_searchbox
def searchbox_film_title(searchterm: str) -> list:
    if not searchterm:
        return []

    results = df_films[df_films["TITLE_ALL"].str.contains(searchterm, case=False, na=False)].head(10)

    return [(f"{row['TITLE_ORIGINAL']} ({int(row['YEAR'])})", row["ID_FILM"]) for _, row in results.iterrows()]





def ids_to_names(ids_string, mapping_table):
#   Fonction qui prend les IDs de personnes et retourne leur véritable nom
    if not isinstance(ids_string, str) or ids_string.lower() == "unknown" or ids_string.strip() == "":
        return ""
    ids_list = [id.strip() for id in ids_string.split("|") if id.strip()]
    names = [mapping_table.get(id, "") for id in ids_list]
    names = [name for name in names if name]
    names = list(dict.fromkeys(names))  # enlever doublons
    return ", ".join(names)







def render_card(movie_row):
#   Affiche le film sous forme de carte
    col_gauche_poster, col_droite_text = st.columns([1, 3], gap="large")

    with col_gauche_poster:
        poster = movie_row.get("POSTER_PATH", "")
        if pd.notna(poster) and str(poster).strip():
            st.image(str(poster), width="stretch")
        else:
            st.caption("🎞️ Pas d'affiche")


    with col_droite_text:

        # Titre principal + année
        title_origin = movie_row.get("TITLE_ORIGINAL", "Titre inconnu")
        title_origin = str(title_origin)
        title_fr = movie_row.get("TITLE_FR", "")
        title_fr = str(title_fr)
        title_fr_main = title_fr.split("|")[0].strip()
        title_en = movie_row.get("TITLE_EN", "")
        title_en = str(title_en)
        title_en_main = title_en.split("|")[0].strip()
        year = movie_row.get("YEAR", "")

        if title_fr_main:
            main_title = f"**{title_fr_main}**"
        elif title_en_main:
            main_title = f"**{title_en_main}**"
        else:
            main_title = f"**{title_origin}**"

        if pd.notna(year) and str(year).strip():
            try:
                main_title += f" ({int(float(year))})"
            except Exception:
                pass
        
        st.markdown(main_title)


        # affichage du titre d'origine si différent du titre FR ou EN
        if title_fr_main != "" and title_en_main != "" and pd.notna(title_origin) and title_origin.strip() != title_fr_main and title_origin.strip() != title_en_main:
            st.caption(
                f"Titre d'origine : '{shorten(title_origin, 60)}'"
                )


        # affichage des genres
        genres = movie_row.get("GENRES", "")
        if pd.notna(genres) and str(genres).strip():
            genres_list = genre_split(genres)
            display_genres = ", ".join(genres_list)
            st.caption(shorten(display_genres, 90))


        # affichage de la note
        # On vérifie d'abord qu'il y a une valeur dans SOURCE_TO_KEEP
        # avant d'afficher une note
        # car un 0 a été mis par défaut dans BEST_RATING pour le ML
        rating = movie_row.get("BEST_RATING", None)
        rating_source = movie_row.get("SOURCE_TO_KEEP", None)
        
        if rating_source is not None and pd.notna(rating_source) and str(rating_source).strip():
            try:
                rating_value = float(rating)
                st.caption(f"⭐ {rating_value:.1f} • 🗳️ {rating_source}")
            except Exception:
                pass


        # affichage du résumé
        summary = movie_row.get("SUMMARY", "")
        if pd.notna(summary) and str(summary).strip():
            st.write(shorten(summary, 280))
        else:
            st.caption("Pas de résumé disponible.")


        # affichage du réalisateur
        director_name = movie_row.get("DIRECTOR", None)
        if pd.notna(director_name) and str(director_name).strip():
            director_list = ids_to_names(director_name, mapping_table_id_to_name)
            st.write(f"Réalisateur(s): {shorten(director_list, 280)}")


        # affichage des acteurs
        actor_name = movie_row.get("ACTOR", None)
        if pd.notna(actor_name) and str(actor_name).strip():
            actor_list = ids_to_names(actor_name, mapping_table_id_to_name)
            st.write(f"Acteur(s): {shorten(actor_list, 280)}")

        # affichage des compositeurs
        composer_name = movie_row.get("COMPOSER", None)
        if pd.notna(composer_name) and str(composer_name).strip():
            composer_list = ids_to_names(composer_name, mapping_table_id_to_name)
            st.write(f"Compositeur(s): {shorten(composer_list, 280)}")


def recommended_render_cards(movies_df):
#   Affiche les films recommandés du dataframe df sous forme de cartes
    for _, row in movies_df.iterrows():
        render_card(row)
        st.divider()



def film_search(selected_film):
    if not selected_film:
        pass

    with st.spinner("Recherche en cours…"):
        # Utilisation de la fonction de recommandation ML
        searched_film, film_to_display = recommend_movies(selected_film, desired_number_of_recommendations)
        
        # Helper : Retourne le df du film recherché
        st.dataframe(
            searched_film,
            width='stretch'
            )

        # Affiche la carte du film recherché
        st.write("Film recherché :")
        render_card(searched_film.iloc[0])


        # Helper : Retourne le df des films recommandés
        st.dataframe(
            film_to_display,
            width='stretch'
            )
        
        # Affiche les cartes des films recommandés
        st.write("Nos recmmandations :")
        recommended_render_cards(film_to_display)



# ----------------------------
# Affichage des titres uniques et des doublons
# ----------------------------
film_title_year = df_films[["TITLE_ALL", "YEAR"]].astype(str).apply(" - ".join, axis=1)

st.write(
    "Liste des titres uniques dans la base de données :"
    )
st.dataframe(
    sorted(film_title_year),
    width='stretch'
    )

st.write(
    "Liste des titres d'origine en doublon dans la base de données :"
    )
st.dataframe(
    df_films[df_films['TITLE_ORIGINAL'].duplicated()][['TITLE_ORIGINAL']],
    width='stretch'
)




# ----------------------------
# Affichage des modes de recherche
# ----------------------------
radio_mode = st.radio(
    "Mode de recherche",
    ["Film", "Acteur", "Réalisateur", "Compositeur"],
    horizontal = True
    )

desired_number_of_recommendations = 5

if radio_mode == "Film":
    input_caption = "Entrez un nom de film. Nous vous recommanderons 5 films proches de celui-ci, selon de nombreux critères."
    input_placeholder = "Inception • Batman • Harry Potter"

    st.caption(input_caption)
    # user_query = st.text_input("🎬 Rechercher un film", placeholder = input_placeholder)

    # Barre de recherche avec autocomplétion
    selected_film = st_searchbox(
    searchbox_film_title,
    placeholder = input_placeholder,
    key = "my_key"
    )


    # cherche les films correspondant à la saisie
#     options = []
#     if user_query:
#         options = df_films[
#             df_films["TITLE_ORIGINAL"]
#             .str.contains(user_query, case=False, na=False)
#         ]["TITLE_ORIGINAL"].head(10).tolist()

    # if pd.notna(year) and str(year).strip():
    #             try:
    #                 main_title += f" ({int(float(year))})"
    #             except Exception:
    #                 pass

    # # propose sélecteur si plusieurs films trouvés avec ce nom
    # if len(options) > 1:
    #     st.caption("Plusieurs films trouvés. Sélectionnez celui que vous voulez.")
    #     selected_movie = st.selectbox("Suggestions", options)
    # else:
    #     selected_movie = options[0] if options else None
    

    # lance la recherche de film avec le film sélectionné
    if selected_film:
        film_search(selected_film)
        

elif radio_mode == "Acteur":
    input_caption = "Entrez un nom d'acteur ou d'actrice. Nous afficherons la liste des films de notre liste auxquels il/elle a participé."
    input_placeholder = "Henry Cavill • Angelina Jolie • Brigitte Bardot"
    st.caption(input_caption)

    user_query = st.text_input(
        "Ta recherche",
        placeholder = input_placeholder
        )
    
elif radio_mode == "Réalisateur":
    input_caption = "Entrez un nom de réalisateur ou réalisatrice. Nous afficherons la liste des films de notre liste auxquels il/elle a participé."
    input_placeholder = "Christopher Nolan • Jean-Luc Godard • Alfred Hitchcock"
    st.caption(input_caption)

    user_query = st.text_input(
        "Ta recherche",
        placeholder = input_placeholder
        )    
    
elif radio_mode == "Compositeur":
    input_caption = "Entrez un nom de compositeur ou compositrice. Nous afficherons la liste des films de notre liste auxquels il/elle a participé."
    input_placeholder = "Hans Zimmer • Ennio Morricone • John Williams"
    st.caption(input_caption)

    user_query = st.text_input(
        "Ta recherche",
        placeholder = input_placeholder
        )














# =============================
# Filtre genre
# =============================
# filter_genres = st.multiselect(
#     "Genres",
#     genre_split(df_films['GENRES'])
#     )


# if filter_genres: # si un filtre est sélectionné, afficher le df filtré 
#     filter_operator = st.radio(
#         "Opérateur",
#         ("or", "and"),
#         horizontal=True
#         )

#     if filter_operator == "or": # Au moins un genre sélectionné
#         pattern = "|".join(filter_genres)
#         mask = df_films["GENRES"].str.contains(pattern, case=False, na=False)

#     elif filter_operator == "and": # Tous les genres sélectionnés doivent être listés
#         mask = df_films["GENRES"].apply(
#             lambda x: all(genre.lower() in x.lower() for genre in filter_genres)
#         )

#     df_films_to_display = df_films[mask]
# else: # si aucun filtre sélectionné, afficher le df entier
#     df_films_to_display = df_films

# st.dataframe(
#     df_films_to_display,
#     width='stretch'
#     )


