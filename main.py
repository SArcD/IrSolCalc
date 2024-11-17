import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
    """Calcula la elevación y azimut solar en grados."""
    # Elevación solar
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    elevation = math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

    # Azimut solar
    cos_azimuth = (math.sin(math.radians(declination)) - 
                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))
    azimuth = math.degrees(math.acos(cos_azimuth)) if elevation > 0 else 0

    # Ajustar el azimut para la tarde
    if hour_angle > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

def generate_solar_path(latitude, fixed_hour):
    """Genera los datos para azimut vs elevación solar."""
    days_of_year = np.arange(1, 366)  # Días del año
    elevations = []
    azimuths = []
    days = []

    for day in days_of_year:
        declination = calculate_declination(day)
        eot = calculate_equation_of_time(day)  # Ecuación del tiempo
        hour_angle = calculate_hour_angle(fixed_hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation > 0:  # Ignorar valores negativos de elevación (noche)
            elevations.append(elevation)
            azimuths.append(azimuth)
            days.append(day)

    return pd.DataFrame({"Día del Año": days, "Azimut (°)": azimuths, "Elevación Solar (°)": elevations})

# Configuración de Streamlit
st.title("Cálculo de Azimut y Elevación Solar")
st.sidebar.header("Parámetros de Entrada")

# Entrada del usuario
latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43)
fixed_hour = st.sidebar.slider("Hora Fija (24h)", 0.0, 24.0, 12.0)

# Generar datos
df = generate_solar_path(latitude, fixed_hour)

# Gráfica interactiva
st.write(f"**Gráfica de Azimut vs Elevación Solar** para Latitud {latitude}° y Hora Fija {fixed_hour}:00")
fig = px.scatter(
    df,
    x="Azimut (°)",
    y="Elevación Solar (°)",
    color="Día del Año",
    color_continuous_scale="Viridis",
    title="Azimut vs Elevación Solar",
    labels={"Día del Año": "Día del Año"}
)
fig.update_layout(
    xaxis_title="Azimut Solar (°)",
    yaxis_title="Elevación Solar (°)",
    coloraxis_colorbar=dict(title="Día del Año"),
    height=600,
    width=900
)
st.plotly_chart(fig)
