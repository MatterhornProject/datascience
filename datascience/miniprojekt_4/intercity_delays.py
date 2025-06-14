import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Analiza opóźnień pociągów", layout="wide")

st.title("Dashboard: Analiza opóźnień pociągów")


img_path = "pendolino.webp"
if os.path.exists(img_path):
    st.image(img_path, caption="Pendolino — symbol nowoczesnych kolei w Polsce", use_container_width=True)
else:
    st.info("Brak zdjęcia 'pendolino.webp' w folderze projektu. Dodaj obrazek do repo, by zobaczyć go tutaj.")


@st.cache_data
def load_data():
    data_path = "delays_cleaned.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        return None

df = load_data()

if df is None:
    st.error("Brak pliku 'delays_cleaned.csv' w repozytorium! Dodaj ten plik i zdeployuj ponownie.")
    st.stop()


st.header("Statystyki ogólne")
col1, col2, col3 = st.columns(3)
col1.metric("Liczba rekordów", f"{len(df):,}")
col2.metric("Średnie opóźnienie [min]", f"{df['delay_minutes'].mean():.2f}")
col3.metric("Mediana opóźnienia [min]", f"{df['delay_minutes'].median():.2f}")

# ---- Histogram opóźnień ----
st.subheader("Rozkład opóźnień (histogram)")
fig, ax = plt.subplots()
ax.hist(df['delay_minutes'], bins=50, edgecolor='black')
ax.set_xlabel("Opóźnienie [min]")
ax.set_ylabel("Liczba przypadków")
st.pyplot(fig)


st.subheader("Średnie opóźnienie wg przewoźnika")
top_n = st.slider("Pokaż top N przewoźników:", 3, 15, 7)
carrier_stats = df.groupby('carrier')['delay_minutes'].mean().sort_values(ascending=False).head(top_n)
st.bar_chart(carrier_stats)


st.subheader("Średnie opóźnienie wg godziny")
if 'hour' in df.columns:
    hourly = df.groupby('hour')['delay_minutes'].mean()
    st.line_chart(hourly)
else:
    st.warning("Brak kolumny 'hour' w danych.")


st.subheader("Średnie opóźnienie wg dnia tygodnia")
dni = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Nd']
if 'dayofweek' in df.columns:
    dow = df.groupby('dayofweek')['delay_minutes'].mean()
    try:
        dow.index = dni[:len(dow)]
    except Exception:
        pass
    st.bar_chart(dow)
else:
    st.warning("Brak kolumny 'dayofweek' w danych.")


st.header("Szczegółowa analiza (wybierz przewoźnika i godzinę)")
if 'carrier' in df.columns and 'hour' in df.columns:
    carrier_options = df['carrier'].dropna().unique()
    selected_carrier = st.selectbox("Przewoźnik:", sorted(carrier_options))
    selected_hour = st.slider("Godzina:", int(df['hour'].min()), int(df['hour'].max()), int(df['hour'].min()))
    filtered = df[(df['carrier'] == selected_carrier) & (df['hour'] == selected_hour)]

    st.write(f"Liczba przypadków: {len(filtered)}")
    st.write(f"Średnie opóźnienie: {filtered['delay_minutes'].mean():.2f} min")

    if not filtered.empty:
        st.subheader("Histogram opóźnień (filtrowany)")
        fig2, ax2 = plt.subplots()
        ax2.hist(filtered['delay_minutes'], bins=30, edgecolor='black')
        ax2.set_xlabel("Opóźnienie [min]")
        ax2.set_ylabel("Liczba przypadków")
        st.pyplot(fig2)
else:
    st.info("Brak kolumn 'carrier' lub 'hour' w danych — ta sekcja jest niedostępna.")


st.header("Podgląd danych (losowa próbka)")
st.dataframe(df.sample(100))

st.markdown("""
---
App by Matterhorn | Powered by Streamlit 🚄
""")
