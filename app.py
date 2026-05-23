import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
st.set_page_config(
    page_title="Dashboard Dataset",
    layout="wide"
)

# LOAD DATA
df = pd.read_csv("dataset_bersih.csv")

# TITLE
st.title("Dashboard Analisis Dataset")
st.write("Visualisasi dataset hasil preprocessing dan augmentasi.")

# SIDEBAR
st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview",
        "Distribusi Data",
        "Heatmap Korelasi",
        "Distribusi Label"
    ]
)

# OVERVIEW
if menu == "Overview":

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Missing Value", df.isnull().sum().sum())

    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

# DISTRIBUSI DATA
elif menu == "Distribusi Data":

    numeric_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    selected_col = st.selectbox(
        "Pilih Kolom Numerik",
        numeric_cols
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df[selected_col],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

# HEATMAP
elif menu == "Heatmap Korelasi":

    numeric_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# DISTRIBUSI LABEL
elif menu == "Distribusi Label":

    categorical_cols = df.select_dtypes(
        include=['object']
    ).columns

    if len(categorical_cols) > 0:

        selected_cat = st.selectbox(
            "Pilih Kolom Kategori",
            categorical_cols
        )

        fig, ax = plt.subplots()

        df[selected_cat].value_counts().plot(
            kind='bar',
            ax=ax
        )

        st.pyplot(fig)

    else:
        st.warning("Tidak ada kolom kategori.")