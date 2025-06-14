import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Dashboard: Analiza opóźnień pociągów")

st.image(
    r"D:\Dev-Env\sources\repos\DataScience\datascience\datascience\miniprojekt_4\pendolino.webp",
    caption="Pendolino — symbol nowoczesnych kolei w Polsce",
    use_container_width=True
)



@st.cache_data
def load_data():
    return pd.read_csv("delays_cleaned.csv")

df = load_data()

st.title("Dashboard: Analiza opóźnień pociągów")


st.header("Statystyki ogólne")
col1, col2, col3 = st.columns(3)
col1.metric("Liczba rekordów", f"{len(df):,}")
col2.metric("Średnie opóźnienie [min]", f"{df['delay_minutes'].mean():.2f}")
col3.metric("Mediana opóźnienia [min]", f"{df['delay_minutes'].median():.2f}")


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
hourly = df.groupby('hour')['delay_minutes'].mean()
st.line_chart(hourly)


st.subheader("Średnie opóźnienie wg dnia tygodnia")
dni = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Nd']
dow = df.groupby('dayofweek')['delay_minutes'].mean()
dow.index = dni[:len(dow)]
st.bar_chart(dow)


st.header("Szczegółowa analiza (wybierz przewoźnika i godzinę)")
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


st.header("Podgląd danych")
st.dataframe(df.sample(100))

st.markdown("""
---
App by [Twoje Imię] | Powered by Streamlit 🚄
""")
