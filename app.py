# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from scripts.gpx_parser import GPXParser
from scripts.elevation_profile import ElevationProfile
from scripts.climb_classification import classify_climb_difficulty

# =====================
# Konfiguracja strony
# =====================
st.set_page_config(page_title="Analiza trasy GPX", layout="wide", page_icon="🚴")
st.title("🚴 Analiza trasy GPX")

# =====================
# Funkcje pomocnicze
# =====================
def build_base_map(track_df: pd.DataFrame, color="blue"):
    start_coords = (track_df["latitude"].iloc[0], track_df["longitude"].iloc[0])
    m = folium.Map(location=start_coords, zoom_start=13, control_scale=True)
    coords = track_df[["latitude", "longitude"]].values.tolist()
    folium.PolyLine(coords, color=color, weight=3, opacity=0.9).add_to(m)
    folium.Marker(coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], popup="Meta", icon=folium.Icon(color="red")).add_to(m)
    return m

# =====================
# Panel boczny (ustawienia)
# =====================
with st.sidebar:
    st.header("Ustawienia analizy")
    smooth_window = st.number_input("Okno wygładzania profilu (rolling mean)", min_value=1, max_value=51, value=5, step=2)
    st.markdown("---")
    st.subheader("Wykrywanie podjazdów")
    min_length = st.number_input("Minimalna długość podjazdu [m]", min_value=100, max_value=20000, value=500, step=100)
    min_gain = st.number_input("Minimalny wznios [m]", min_value=10, max_value=2000, value=30, step=10)
    min_avg_slope = st.number_input("Minimalne średnie nachylenie (%)", min_value=1.0, max_value=15.0, value=2.0, step=0.5)
    st.markdown("---")
    st.subheader("Przedziały nachylenia (dla tabeli)")
    slope_thresholds_str = st.text_input("Progi (w %), rozdzielone przecinkami", value="2,4,5,8")
    try:
        slope_thresholds = tuple(float(x.strip()) for x in slope_thresholds_str.split(",") if x.strip() != "")
    except Exception:
        slope_thresholds = (2, 4, 5, 8)
        st.warning("Nieprawidłowy format progów – używam domyślnych: 2,4,5,8")

# =====================
# Upload pliku GPX
# =====================
uploaded_file = st.file_uploader("Wgraj plik GPX", type="gpx")
if uploaded_file is None:
    st.info("Wgraj plik GPX, aby rozpocząć analizę.")
    st.stop()

# =====================
# Parsowanie i statystyki
# =====================
parser = GPXParser(uploaded_file.read())
track_df = parser.parse_to_dataframe()

total_distance_km = float(track_df["km"].max())
total_ascent_m = float(parser.get_total_ascent(smooth_window=smooth_window))
total_descent_m = float(parser.get_total_descent(smooth_window=smooth_window))

st.subheader("📊 Statystyki trasy")
c1, c2, c3 = st.columns(3)
c1.metric("Dystans", f"{total_distance_km:.2f} km")
c2.metric("Całkowity wznios", f"{total_ascent_m:.0f} m")
c3.metric("Całkowity spadek", f"{total_descent_m:.0f} m")

# =====================
# Mapa bazowa
# =====================
st.subheader("🗺️ Mapa trasy")
base_map = build_base_map(track_df, color="blue")

# =====================
# Analiza profilu głównego
# =====================
st.subheader("📈 Profil wysokościowy (cała trasa)")
main_profile = ElevationProfile(track_df, seg_unit_km=0.2)
fig, ax = main_profile.plot(
    show_labels=False,
    show_background=True,
    smooth_window=smooth_window,
    slope_thresholds=slope_thresholds
)
st.pyplot(fig, use_container_width=True)

# =====================
# Tabela długości wg przedziałów nachylenia
# =====================
st.subheader("📋 Długość odcinków w przedziałach nachylenia")
slope_df = main_profile.compute_slope_lengths(
    smooth_window=smooth_window,
    slope_thresholds=slope_thresholds
)
st.dataframe(slope_df, use_container_width=True)

# =====================
# Wykrywanie podjazdów
# =====================
st.subheader("🏔️ Wykryte podjazdy")
climbs_df = main_profile.detect_climbs(
    min_length_m=min_length,
    min_gain_m=min_gain,
    smooth_window=smooth_window,
    min_avg_slope=min_avg_slope
)

if climbs_df.empty:
    st.warning("Nie wykryto podjazdów dla podanych parametrów.")
else:
    st.dataframe(climbs_df[['start_km', 'end_km', 'length_m', 'gain_m', 'avg_grade_pct']], use_container_width=True)

    for number, row in climbs_df.iterrows():
        popup_text = (
            f"Podjazd {number + 1}<br>"
            f"Długość: {row['length_m']} m<br>"
            f"Wznios: +{row['gain_m']} m<br>"
            f"Śr. nachylenie: {row['avg_grade_pct']}%"
        )
        folium.Marker(
            [row["start_lat"], row["start_lon"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color="orange", icon="arrow-up")
        ).add_to(base_map)
        folium.Marker(
            [row["end_lat"], row["end_lon"]],
            popup=f"Koniec podjazdu {number + 1}",
            icon=folium.Icon(color="blue", icon="flag")
        ).add_to(base_map)

    st.subheader("🗺️ Mapa z podjazdami")
    st_folium(base_map, width=1000, height=550)

    st.subheader("📊 Profile wysokości podjazdów")
    tab_titles = [f"Podjazd {i+1}" for i in range(len(climbs_df))]
    tabs = st.tabs(tab_titles)

    for i, row in enumerate(climbs_df.itertuples()):
        with tabs[i]:
            climb_df = track_df.iloc[
                track_df['km'].sub(row.start_km).abs().idxmin() :
                track_df['km'].sub(row.end_km).abs().idxmin() + 1
            ].reset_index(drop=True)

            if len(climb_df) < 3:
                st.info("Zbyt mało punktów do sensownego wykresu. Spróbuj zmienić wygładzanie lub progi.")
            else:
                climb_df = climb_df.copy()
                climb_df["km"] -= climb_df["km"].iloc[0]
                climb_profile = ElevationProfile(climb_df, seg_unit_km=0.1)
                fig_c, ax_c = climb_profile.plot(
                    show_labels=False,
                    show_background=False,
                    smooth_window=smooth_window,
                    slope_thresholds=slope_thresholds
                )
                ax_c.set_ylim(climb_df["elevation"].min(), climb_df["elevation"].max())

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Długość", f"{row.length_m} m")
                    st.caption(f"**Od** {row.start_km:.2f} km **do** {row.end_km:.2f} km trasy.")
                    st.metric("Wznios", f"{row.gain_m} m")
                    st.metric("Śr. nachylenie", f"{row.avg_grade_pct} %")
                    st.metric("Kategoria", classify_climb_difficulty(row.length_m, row.avg_grade_pct)[0])

                with col2:
                    st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
 

# =====================
# Stopka
# =====================
st.markdown("---")
st.caption("© Aplikacja demonstracyjna – analiza GPX w Streamlit. Ustawienia wykrywania podjazdów: długość i wznios w panelu bocznym.")
