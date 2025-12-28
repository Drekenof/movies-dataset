# =============================
# IMPORT LIBRAIRIE
# =============================
 
import streamlit as st
import pandas as pd

# from reco_engine import (
#     recommend_movies
#     )

# =============================
# LOAD DATA
# =============================
# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    # df = pd.read_csv("data/movies_genres_summary.csv")
    df_films = pd.read_csv("DIM_FILM_LIST_FINAL_short_for_quick_work.csv",
    nrows = 1000,
    encoding='latin1')
    return df_films

df_films = load_data()



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

if st.button("Rechercher"):
    try:
        user_query = user_query.strip()
        if not user_query:
            st.warning("Entrez un texte.")
            st.stop()

        with st.spinner("Recherche en cours…"):
            if radio_mode == "Film" :
                df = recommend_by_title(user_query, n = N_RECO, sort_by = sort_mode)
            elif radio_mode == "Acteur" :
                df = films_by_actor(user_query, n = N_RECO, sort_by = sort_mode)
            elif radio_mode == "Réalisateur" :
                df = films_by_director(user_query, n = N_RECO, sort_by = sort_mode)
            elif radio_mode == "Compositeur" :
                df = films_by_composer(user_query, n = N_RECO, sort_by = sort_mode)

        st.success(f"✅ {mode} — {N_RECO} résultats")
        render_cards(df_films, k = N_RECO)

    except Exception as e:
        st.error(str(e))


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
    use_container_width=True
    )