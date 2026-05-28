import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# CONFIG
st.set_page_config(
    page_title="Dashboard EDA - Student Loan",
    layout="wide"
)

# Warna
COLOR_LAYAK = '#2E86AB'
COLOR_TIDAK = '#E84855'
PALETTE = {'Layak': COLOR_LAYAK, 'Tidak Layak': COLOR_TIDAK}

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_final17_label.csv")
    df['Label_enc'] = (df['Label'] == 'Layak').astype(int)
    return df

df = load_data()

# TITLE
st.title("Dashboard EDA - Student Loan Eligibility")
st.write("Visualisasi dan analisis dataset kelayakan pinjaman mahasiswa.")

# SIDEBAR
st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview",
        "Distribusi Data",
        "Heatmap Korelasi",
        "Distribusi Label",
        "Fitur Kategorikal",
        "Fitur Numerik",
        "Insight"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filter Data**")

label_filter = st.sidebar.multiselect(
    "Label",
    options=df['Label'].unique().tolist(),
    default=df['Label'].unique().tolist()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df['Gender'].unique().tolist(),
    default=df['Gender'].unique().tolist()
)

working_filter = st.sidebar.multiselect(
    "Working_Student",
    options=df['Working_Student'].unique().tolist(),
    default=df['Working_Student'].unique().tolist()
)

# Terapkan filter
df_filtered = df[
    (df['Label'].isin(label_filter)) &
    (df['Gender'].isin(gender_filter)) &
    (df['Working_Student'].isin(working_filter))
]

st.sidebar.markdown(f"**Total data:** {len(df_filtered):,} baris")

# =============================================================================
# OVERVIEW
# =============================================================================
if menu == "Overview":

    st.subheader("Preview Dataset")
    st.dataframe(df_filtered.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df_filtered.shape[0])
    col2.metric("Jumlah Kolom", df_filtered.shape[1])
    col3.metric("Missing Value", df_filtered.isnull().sum().sum())

    st.subheader("Statistik Deskriptif")
    st.write(df_filtered.describe())

# =============================================================================
# DISTRIBUSI DATA
# =============================================================================
elif menu == "Distribusi Data":

    st.subheader("Distribusi Data")

    numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Label_enc']

    selected_col = st.selectbox("Pilih Kolom Numerik", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(df_filtered[selected_col], kde=True, ax=ax, color=COLOR_LAYAK)
    ax.set_xlabel(selected_col)
    ax.set_ylabel("Jumlah")
    sns.despine()
    st.pyplot(fig)
    plt.close()

# =============================================================================
# HEATMAP KORELASI
# =============================================================================
elif menu == "Heatmap Korelasi":

    st.subheader("Heatmap Korelasi")

    num_cols_corr = ['Age', 'Payment_History', 'Parental_Income_IDR_Monthly',
                     'Loan_Amount_IDR', 'Course_Credits', 'Liability',
                     'Attendance', 'Grade_Average', 'Label_enc']

    corr = df_filtered[num_cols_corr].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.5, annot_kws={'size': 9})
    ax.set_title('Label_enc: 1=Layak, 0=Tidak Layak', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**Korelasi terhadap Label**")
    corr_label = corr['Label_enc'].drop('Label_enc').sort_values(ascending=False).reset_index()
    corr_label.columns = ['Fitur', 'Korelasi']
    corr_label['Keterangan'] = corr_label['Korelasi'].apply(
        lambda x: 'Positif kuat' if x > 0.3
        else ('Positif sedang' if x > 0.1
        else ('Positif lemah' if x > 0
        else ('Negatif sedang' if x < -0.1 else 'Negatif lemah')))
    )
    st.dataframe(corr_label.round(3), use_container_width=True)

# =============================================================================
# DISTRIBUSI LABEL
# =============================================================================
elif menu == "Distribusi Label":

    st.subheader("Distribusi Label")

    categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()

    if len(categorical_cols) > 0:

        selected_cat = st.selectbox("Pilih Kolom Kategori", categorical_cols)

        counts = df_filtered[selected_cat].value_counts()

        fig, ax = plt.subplots()
        bars = ax.bar(counts.index, counts.values,
                      color=COLOR_LAYAK, edgecolor='white')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max() * 0.02,
                    f'{val:,}', ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('Jumlah')
        ax.set_ylim(0, counts.max() * 1.2)
        ax.tick_params(axis='x', rotation=30)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.warning("Tidak ada kolom kategori.")

# =============================================================================
# FITUR KATEGORIKAL
# =============================================================================
elif menu == "Fitur Kategorikal":

    st.subheader("Fitur Kategorikal vs Label")
    st.write("Perbandingan setiap fitur kategorikal terhadap Label (%).")

    cat_feats = ['Gender', 'Home_Ownership', 'Loan_Purpose',
                 'Previous_Loan', 'Working_Student', 'Residence_Type']

    col1, col2 = st.columns(2)
    cols = [col1, col2]

    for i, feat in enumerate(cat_feats):
        with cols[i % 2]:
            st.markdown(f"**{feat}**")
            ct = pd.crosstab(df_filtered[feat], df_filtered['Label'],
                             normalize='index') * 100
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ct.plot(kind='bar', ax=ax, color=[COLOR_LAYAK, COLOR_TIDAK],
                    edgecolor='white', linewidth=1.2)
            ax.set_ylabel('Persentase (%)')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=30)
            ax.legend(['Layak', 'Tidak Layak'], fontsize=8)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', fontsize=7, padding=2)
            ax.set_ylim(0, 100)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.markdown("**Parent_Job - Persentase Layak**")
    pj_rate = df_filtered.groupby('Parent_Job')['Label'].apply(
        lambda x: (x == 'Layak').mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors_pj = [COLOR_LAYAK if v > 50 else COLOR_TIDAK for v in pj_rate.values]
    bars = ax.bar(pj_rate.index, pj_rate.values, color=colors_pj, edgecolor='white')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='Threshold 50%')
    ax.set_ylabel('% Layak')
    ax.set_ylim(0, 90)
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, pj_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9)
    ax.legend()
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# =============================================================================
# FITUR NUMERIK
# =============================================================================
elif menu == "Fitur Numerik":

    st.subheader("Fitur Numerik vs Label")
    st.write("Rata-rata setiap fitur numerik berdasarkan Label.")

    num_features = ['Payment_History', 'Parental_Income_IDR_Monthly', 'Loan_Amount_IDR',
                    'Grade_Average', 'Attendance', 'Course_Credits', 'Liability', 'Age']

    avg = df_filtered.groupby('Label')[num_features].mean()

    col1, col2 = st.columns(2)
    cols = [col1, col2]

    for i, col in enumerate(num_features):
        with cols[i % 2]:
            st.markdown(f"**{col}**")
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(avg.index, avg[col],
                          color=[COLOR_LAYAK, COLOR_TIDAK], edgecolor='white')
            ax.set_ylabel('Rata-rata')
            ax.set_ylim(0, avg[col].max() * 1.25)
            for bar, val in zip(bars, avg[col]):
                if col in ['Parental_Income_IDR_Monthly', 'Loan_Amount_IDR']:
                    label_text = f'{val/1e6:.1f}M'
                else:
                    label_text = f'{val:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + avg[col].max() * 0.02,
                        label_text, ha='center', fontsize=10, fontweight='bold')
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.markdown("**Tabel Rata-rata per Label**")
    avg_display = avg.copy()
    avg_display['Parental_Income_IDR_Monthly'] = avg_display['Parental_Income_IDR_Monthly'].apply(
        lambda x: f'Rp {x/1e6:.2f}M')
    avg_display['Loan_Amount_IDR'] = avg_display['Loan_Amount_IDR'].apply(
        lambda x: f'Rp {x/1e6:.2f}M')
    st.dataframe(avg_display.round(2), use_container_width=True)

# =============================================================================
# INSIGHT
# =============================================================================
elif menu == "Insight":

    st.subheader("Insight dan Kesimpulan")

    st.markdown("### Pertanyaan 1: Apakah Working_Student memengaruhi kelayakan?")
    ws_pct = pd.crosstab(df_filtered['Working_Student'], df_filtered['Label'],
                          normalize='index') * 100
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ws_pct.plot(kind='barh', ax=ax, color=[COLOR_LAYAK, COLOR_TIDAK], edgecolor='white')
    ax.set_xlabel('Persentase (%)')
    ax.set_ylabel('')
    ax.legend(['Layak', 'Tidak Layak'])
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3)
    ax.set_xlim(0, 80)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("Mahasiswa Bekerja_Karena_Butuh memiliki tingkat kelayakan lebih rendah (~45%) dibanding Bekerja_Optional (~53%).")

    st.markdown("---")
    st.markdown("### Pertanyaan 2: Apakah Parental_Income_IDR_Monthly memengaruhi kelayakan?")
    df_temp = df_filtered.copy()
    df_temp['Income_Group'] = pd.cut(df_temp['Parental_Income_IDR_Monthly'],
        bins=[0, 15e6, 30e6, 50e6, 100e6],
        labels=['< 15 Juta', '15 - 30 Juta', '30 - 50 Juta', '> 50 Juta'])
    ig_rate = df_temp.groupby('Income_Group', observed=True)['Label_enc'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_ig = [COLOR_LAYAK if v > 50 else COLOR_TIDAK for v in ig_rate.values]
    bars = ax.bar(ig_rate.index, ig_rate.values, color=colors_ig, edgecolor='white')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='Threshold 50%')
    ax.set_ylabel('% Layak')
    ax.set_ylim(0, 85)
    for bar, val in zip(bars, ig_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.legend()
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("Semakin tinggi pendapatan orang tua, semakin besar kemungkinan mahasiswa dinyatakan Layak.")

    st.markdown("---")
    st.markdown("### Pertanyaan 3: Apakah Liability memengaruhi kelayakan?")
    liab_rate = df_filtered.groupby('Liability')['Label_enc'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(liab_rate.index, liab_rate.values,
           color=[COLOR_LAYAK if v > 50 else COLOR_TIDAK for v in liab_rate.values],
           edgecolor='white')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Liability')
    ax.set_ylabel('% Layak')
    ax.set_ylim(0, 90)
    for i, val in enumerate(liab_rate.values):
        ax.text(liab_rate.index[i], val + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("Semakin banyak tanggungan, semakin rendah tingkat kelayakan mahasiswa.")

    st.markdown("---")
    st.markdown("### Kesimpulan Utama")
    st.markdown("""
    | Fitur | Korelasi | Kesimpulan |
    |---|---|---|
    | Payment_History | 0.557 | Fitur paling berpengaruh terhadap kelayakan |
    | Parental_Income_IDR_Monthly | 0.330 | Pendapatan orang tua tinggi meningkatkan peluang Layak |
    | Liability | -0.203 | Semakin banyak tanggungan, semakin kecil peluang Layak |
    | Grade_Average | 0.141 | Berpengaruh kecil |
    | Attendance | 0.115 | Berpengaruh kecil |
    | Loan_Amount_IDR | 0.058 | Hampir tidak berpengaruh |
    | Age | 0.050 | Hampir tidak berpengaruh |
    | Loan_Int_Rate | - | Nilai konstan, tidak informatif, sebaiknya dihapus sebelum modeling |
    """)
