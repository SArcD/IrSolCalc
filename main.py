import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Funciones de cálculo
def calculate_declination(day_of_year):
    """Calcula la declinación solar en función del día del año."""
    return 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

def calculate_equation_of_time(day_of_year):
    """Calcula la ecuación del tiempo en minutos."""
    B = math.radians((360 / 365) * (day_of_year - 81))
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

def calculate_hour_angle(hour, equation_of_time):
    """Corrige el ángulo horario por la ecuación del tiempo."""
    solar_time = hour + (equation_of_time / 60)
    return 15 * (solar_time - 12)

def calculate_solar_position(latitude, declination, hour_angle):
    """Calcula la elevación solar y el azimut en grados."""
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    if sin_altitude <= 0:
        return None, None  # El sol está debajo del horizonte

    elevation = math.degrees(math.asin(sin_altitude))

    cos_azimuth = (math.sin(math.radians(declination)) -
                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))

    azimuth = math.degrees(math.acos(cos_azimuth)) if cos_azimuth <= 1 else 0
    if hour_angle > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

def generate_daily_solar_position(latitude, day_of_year):
    """Genera los datos de posición solar para todas las horas del día."""
    hours = np.arange(0, 24, 0.5)
    elevations, azimuths, hours_list = [], [], []

    declination = calculate_declination(day_of_year)
    eot = calculate_equation_of_time(day_of_year)

    for hour in hours:
        hour_angle = calculate_hour_angle(hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation is not None:
            elevations.append(elevation)
            azimuths.append(azimuth)
            hours_list.append(hour)

    return pd.DataFrame({
        "Hora del Día": hours_list,
        "Elevación Solar (°)": elevations,
        "Azimut Solar (°)": azimuths
    })

def generate_analemma(latitude, fixed_hour):
    """Genera los datos del analema solar."""
    days_of_year = np.arange(1, 366)
    elevations, azimuths, days = [], [], []

    for day in days_of_year:
        declination = calculate_declination(day)
        eot = calculate_equation_of_time(day)
        hour_angle = calculate_hour_angle(fixed_hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation is not None:
            elevations.append(elevation)
            azimuths.append(azimuth)
            days.append(day)

    return pd.DataFrame({
        "Día del Año": days,
        "Azimut Solar (°)": azimuths,
        "Elevación Solar (°)": elevations
    })

# Configuración de Streamlit
st.title("Gráficas de Posición Solar y Analema")

# Barra lateral para los inputs
st.sidebar.header("Parámetros de Entrada")
latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)
day_of_year = st.sidebar.slider("Día del Año", 1, 365, 172)
selected_hour = st.sidebar.slider("Hora del Día (24h)", 0.0, 24.0, 12.0, step=0.5)

# Datos para la posición solar y el analema
df_position = generate_daily_solar_position(latitude, day_of_year)
df_analemma = generate_analemma(latitude, selected_hour)

# Transformar posiciones del analema a coordenadas esféricas
analemma_positions = [
    (
        math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim)),
        math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim)),
        math.cos(math.radians(90 - elev))
    )
    for elev, azim in zip(df_analemma["Elevación Solar (°)"], df_analemma["Azimut Solar (°)"])
]

analemma_x, analemma_y, analemma_z = zip(*analemma_positions)

# Gráfica del analema
fig_analemma = go.Figure()

fig_analemma.add_trace(go.Scatter3d(
    x=analemma_x,
    y=analemma_y,
    z=analemma_z,
    mode="markers+lines",
    marker=dict(size=6, color=df_analemma["Día del Año"], colorscale="Viridis", showscale=False),
    name="Trayectoria del Analema"
))

fig_analemma.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z (Elevación)"
    ),
    title="Analema Solar",
    height=700,
    width=900
)

# Gráfica de posición solar
selected_row = df_position[df_position["Hora del Día"] == selected_hour]
if not selected_row.empty:
    elev = selected_row["Elevación Solar (°)"].values[0]
    azim = selected_row["Azimut Solar (°)"].values[0]
else:
    elev = azim = 0

arrow_x = math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim))
arrow_y = math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim))
arrow_z = math.cos(math.radians(90 - elev))

fig_position = go.Figure()

fig_position.add_trace(go.Scatter3d(
    x=[arrow_x],
    y=[arrow_y],
    z=[arrow_z],
    mode="markers",
    marker=dict(size=8, color="blue"),
    name="Posición Solar Actual"
))

fig_position.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z (Elevación)"
    ),
    title="Posición Solar",
    height=700,
    width=900
)

# Tabs para las gráficas
tab1, tab2 = st.tabs(["Posición Solar", "Analema Solar"])

with tab1:
    st.plotly_chart(fig_position, use_container_width=True)

with tab2:
    st.plotly_chart(fig_analemma, use_container_width=True)
