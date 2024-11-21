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
        return None, None

    elevation = math.degrees(math.asin(sin_altitude))

    cos_azimuth = (math.sin(math.radians(declination)) -
                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))
    azimuth = math.degrees(math.acos(cos_azimuth)) if cos_azimuth <= 1 else 0
    if hour_angle > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

def generate_analemma(latitude, fixed_hour):
    """Genera los datos para el analema."""
    days_of_year = np.arange(1, 366)
    elevations = []
    azimuths = []
    days = []

    for day in days_of_year:
        declination = calculate_declination(day)
        eot = calculate_equation_of_time(day)
        hour_angle = calculate_hour_angle(fixed_hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation is not None:
            elevations.append(elevation)
            azimuths.append(azimuth)
            days.append(day)

    return pd.DataFrame({"Día del Año": days, "Azimut (°)": azimuths, "Elevación Solar (°)": elevations})

# Configuración de Streamlit
st.title("Vista del Observador: Posición Solar, Analema y Radiación")

# Barra lateral para los inputs
st.sidebar.header("Parámetros de Entrada")
latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)
day_of_year = st.sidebar.slider("Día del Año", 1, 365, 172)
selected_hour = st.sidebar.slider("Hora del Día (24h)", 0.0, 24.0, 12.0, step=0.5)

# Gráfica de la posición solar
st.subheader("Gráfica de la Posición Solar")
df_position = generate_analemma(latitude, selected_hour)

# Generar analema
df_analemma = generate_analemma(latitude, selected_hour)

# Convertir a coordenadas esféricas
solar_positions = [
    (
        math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim)),
        math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim)),
        math.cos(math.radians(90 - elev))
    )
    for elev, azim in zip(df_analemma["Elevación Solar (°)"], df_analemma["Azimut (°)"])
]

solar_x, solar_y, solar_z = zip(*solar_positions)

# Crear gráfica del analema
fig_analemma = px.scatter_3d(
    df_analemma,
    x=solar_x,
    y=solar_y,
    z=solar_z,
    color="Día del Año",
    title="Gráfica del Analema Solar",
    labels={
        "x": "Azimut (°)",
        "y": "Elevación Solar (°)",
        "z": "Día del Año"
    },
)
fig_analemma.update_traces(marker=dict(size=4))
fig_analemma.update_layout(
    scene=dict(
        xaxis_title="X (Azimut)",
        yaxis_title="Y",
        zaxis_title="Z (Elevación)"
    ),
    height=700,
    width=900
)

# Mostrar gráfica del analema
st.plotly_chart(fig_analemma)

# Sección de radiación solar
st.subheader("Cálculo de Radiación Solar")
transmission_coefficient = st.sidebar.slider("Coeficiente de Transmisión", 0.0, 1.0, 0.75)

def calculate_solar_power(latitude, day_of_year, local_hour, transmission_coefficient):
    S0 = 1361
    declination = calculate_declination(day_of_year)
    solar_hour = local_hour - 12
    hour_angle = 15 * solar_hour
    sin_alpha = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                 math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    if sin_alpha <= 0:
        return 0
    return S0 * transmission_coefficient * sin_alpha

radiation_power = calculate_solar_power(latitude, day_of_year, selected_hour, transmission_coefficient)
st.write(f"La potencia de radiación solar recibida es de aproximadamente **{radiation_power:.2f} W/m²**.")
