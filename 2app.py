import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Indian Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# ==================================================
# THEME
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f1f1f, #666);
}

[data-testid="stSidebar"] {
    background: #111;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: white !important;
}

.stTextInput input,
[data-baseweb="select"] > div {
    background: #ddd !important;
    color: #000 !important;
    border: 2px solid #222 !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: #000 !important;
    color: white !important;
    width: 100%;
    border-radius: 10px;
    border: 2px solid #777 !important;
}

[data-testid="stMetric"] {
    background: #333;
    border-radius: 10px;
    border: 1px solid #777;
}

[data-testid="stMetricLabel"] {
    color: #ccc !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
}

.top {
    background: linear-gradient(135deg, #111, #555);
    color: white;
    padding: 15px;
    margin: 10px 0;
    border-radius: 12px;
    border-left: 5px solid #aaa;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD EXCEL DATASET
# ==================================================

# Folder where 2app.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Excel file path
excel_file = os.path.join(
    base_dir,
    "Indian_Music_Dataset_2000_Tracks.xlsx"
)

# Check Excel file
if not os.path.exists(excel_file):
    st.error(
        "Excel file not found!\n\n"
        "Please put 'Indian_Music_Dataset_2000_Tracks.xlsx' "
        "in the same folder as 2app.py."
    )
    st.stop()

# Read Excel file
df = pd.read_excel(
    excel_file,
    sheet_name="Music Tracks Dataset",
    header=3
)

# ==================================================
# CLEAN DATA
# ==================================================

df = df.dropna(how="all")

df = df.loc[
    :,
    ~df.columns.astype(str).str.startswith("Unnamed")
]

# ==================================================
# RENAME COLUMNS
# ==================================================

df = df.rename(columns={
    "Track ID": "track_id",
    "Song Name": "song_name",
    "Artist": "artist",
    "Album / Movie": "album",
    "Language": "language",
    "Genre": "genre",
    "Release Year": "release_year",
    "Rating (out of 5)": "rating",
    "Streams (Millions)": "streams"
})

# ==================================================
# CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "song_name",
    "artist",
    "album",
    "language",
    "genre",
    "rating"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing columns in Excel file: {missing_columns}"
    )
    st.stop()

# ==================================================
# CONVERT DATA
# ==================================================

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
).fillna(0)

for col in [
    "song_name",
    "artist",
    "album",
    "language",
    "genre"
]:
    df[col] = df[col].fillna("").astype(str)

# ==================================================
# CREATE TAGS
# ==================================================

df["tags"] = (
    df["artist"] + " " +
    df["album"] + " " +
    df["language"] + " " +
    df["genre"]
)

# ==================================================
# SIMILARITY MATRIX
# ==================================================

similarity_file = os.path.join(
    base_dir,
    "indian_music_similarities.pkl"
)

if not os.path.exists(similarity_file):

    cv = CountVectorizer(
        max_features=10000,
        stop_words="english"
    )

    dtm = cv.fit_transform(
        df["tags"]
    )

    similarities = cosine_similarity(dtm)

    with open(similarity_file, "wb") as f:
        pickle.dump(similarities, f)

else:

    with open(similarity_file, "rb") as f:
        similarities = pickle.load(f)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🎧 Music Recommender")

st.sidebar.markdown("---")

feature = st.sidebar.radio(
    "Features",
    [
        "🔥 Top 5 Recommended Songs",
        "🔍 Search Songs",
        "🌍 Language Distribution",
        "📊 Complete Dataset"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    "Made with ❤️"
)

# ==================================================
# TITLE
# ==================================================

st.title(
    "🎵 Indian Music Recommendation System"
)

# ==================================================
# DATASET INFORMATION
# ==================================================

total_songs = len(df)

total_artists = df["artist"].nunique()

total_languages = df["language"].nunique()

# ==================================================
# TOP 5 RECOMMENDED SONGS
# ==================================================

if feature == "🔥 Top 5 Recommended Songs":

    st.subheader(
        "🔥 Top 5 Recommended Songs"
    )

    top5 = df.sort_values(
        "rating",
        ascending=False
    ).head(5)

    for count, song in enumerate(
        top5.itertuples(),
        1
    ):

        st.subheader(
            f"🔥 {count}. {song.song_name}"
        )

        st.write(
            "⭐ Rating:",
            f"{song.rating}/5"
        )

        st.write(
            "👨‍🎤 Artist:",
            song.artist
        )

        st.write(
            "🌍 Language:",
            song.language
        )

        st.divider()

# ==================================================
# SEARCH SONGS
# ==================================================

elif feature == "🔍 Search Songs":

    st.subheader(
        "🔍 Search Songs"
    )

    search = st.text_input(
        "🔍 Search Song"
    )

    names = sorted(
        df["song_name"]
        .dropna()
        .unique()
    )

    if search:

        names = [
            song
            for song in names
            if search.lower() in song.lower()
        ]

    if names:

        name = st.selectbox(
            "🎵 Select a Song",
            names
        )

        # Get selected song
        selected_rows = df[
            df["song_name"] == name
        ]

        song_data = selected_rows.iloc[0]

        # ==================================================
        # SONG INFORMATION
        # ==================================================

        st.subheader(
            "🎵 Song Information"
        )

        st.write(
            "🎵 Song:",
            song_data["song_name"]
        )

        st.write(
            "👨‍🎤 Artist:",
            song_data["artist"]
        )

        st.write(
            "🌍 Language:",
            song_data["language"]
        )

        st.write(
            "🎼 Genre:",
            song_data["genre"]
        )

        st.write(
            "💿 Album / Movie:",
            song_data["album"]
        )

        st.write(
            "⭐ Rating:",
            f"{song_data['rating']}/5"
        )

        st.write(
            "📅 Release Year:",
            song_data["release_year"]
        )

        st.divider()

        # ==================================================
        # RECOMMEND BUTTON
        # ==================================================

        if st.button(
            "🎶 Recommend Songs"
        ):

            # Get original index
            index = selected_rows.index[0]

            # Similarity ranking
            similarity_index = sorted(
                enumerate(
                    similarities[index]
                ),
                key=lambda x: x[1],
                reverse=True
            )

            selected_language = (
                song_data["language"]
            )

            recommended = []

            # ==================================================
            # FIND SAME LANGUAGE SONGS
            # ==================================================

            for i, score in similarity_index:

                # Skip selected song
                if i == index:
                    continue

                song = df.iloc[i]

                if (
                    song["language"]
                    == selected_language
                ):
                    recommended.append(song)

                if len(recommended) == 5:
                    break

            # ==================================================
            # DISPLAY RECOMMENDATIONS
            # ==================================================

            st.subheader(
                "🎧 Recommended Songs"
            )

            if len(recommended) > 0:

                for count, song in enumerate(
                    recommended,
                    1
                ):

                    st.subheader(
                        f"🎵 {count}. {song['song_name']}"
                    )

                    st.write(
                        "⭐ Rating:",
                        f"{song['rating']}/5"
                    )

                    st.write(
                        "👨‍🎤 Artist:",
                        song["artist"]
                    )

                    st.write(
                        "🌍 Language:",
                        song["language"]
                    )

                    st.write(
                        "🎼 Genre:",
                        song["genre"]
                    )

                    st.divider()

            else:

                st.warning(
                    "No recommendations found "
                    "in the same language."
                )

    else:

        st.warning(
            "No song found. "
            "Please try another search."
        )

# ==================================================
# LANGUAGE DISTRIBUTION
# ==================================================

elif feature == "🌍 Language Distribution":

    st.subheader(
        "🌍 Language Distribution"
    )

    language_count = (
        df["language"]
        .value_counts()
    )

    st.bar_chart(
        language_count
    )

    st.markdown("---")

    st.subheader(
        "📋 Language Details"
    )

    language_table = (
        language_count
        .reset_index()
    )

    language_table.columns = [
        "Language",
        "Number of Songs"
    ]

    st.dataframe(
        language_table,
        use_container_width=True
    )

# ==================================================
# COMPLETE DATASET
# ==================================================

elif feature == "📊 Complete Dataset":

    st.subheader(
        "📊 Dataset Statistics"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🎵 Total Songs",
            total_songs
        )

    with col2:

        st.metric(
            "👨‍🎤 Total Artists",
            total_artists
        )

    with col3:

        st.metric(
            "🌍 Languages",
            total_languages
        )

    st.markdown("---")

    st.subheader(
        "📋 Complete Dataset"
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Thank You ❤️"
)
