import math
import numpy as np
import pandas as pd
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
    """Calcula la elevación solar (altitud) en grados."""
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    return math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

def calculate_radiation(altitude):
    """Calcula la radiación solar total incidente en W/m²."""
    S0 = 1361  # Constante solar (W/m²)
    T_a = 0.75  # Transmisión atmosférica promedio
    return S0 * T_a * math.sin(math.radians(altitude)) if altitude > 0 else 0

def calculate_uv_radiation(total_radiation):
    """Calcula la fracción de radiación solar correspondiente a la luz UV."""
    uv_fraction = 0.05  # 5% de la radiación total
    return total_radiation * uv_fraction

def generate_radiation_data(latitude, day_of_year, radiation_type):
    """Genera los datos de radiación para cada hora del día según el tipo de radiación."""
    hours_of_day = np.arange(0, 24, 0.5)  # Horas del día en intervalos de 0.5 horas
    radiations = []
    altitudes = []

    declination = calculate_declination(day_of_year)
    eot = calculate_equation_of_time(day_of_year)  # Ecuación del tiempo

    for hour in hours_of_day:
        hour_angle = calculate_hour_angle(hour, eot)
        altitude = calculate_solar_position(latitude, declination, hour_angle)
        total_radiation = calculate_radiation(altitude)
        
        if radiation_type == "Radiación Total":
            radiation = total_radiation
        elif radiation_type == "Radiación UV":
            radiation = calculate_uv_radiation(total_radiation)
        else:
            radiation = 0  # Valor por defecto

        altitudes.append(altitude)
        radiations.append(radiation)

    return pd.DataFrame({
        "Hora del Día": hours_of_day,
        "Altitud Solar (°)": altitudes,
        "Radiación (W/m²)": radiations
    })

# Configuración de Streamlit
st.title("Calculadora de Radiación Solar")

# Entrada global
latitude = st.number_input("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)

# Sección de radiación total
st.subheader("Radiación Total")
day_of_year_total = st.slider("Día del Año (Radiación Total)", 1, 365, 172)
df_total = generate_radiation_data(latitude, day_of_year_total, "Radiación Total")
fig_total = px.line(
    df_total,
    x="Hora del Día",
    y="Radiación (W/m²)",
    title="Radiación Total durante el Día",
    labels={"Hora del Día": "Hora del Día", "Radiación (W/m²)": "Radiación Total (W/m²)"},
)
st.plotly_chart(fig_total)

# Sección de radiación UV
st.subheader("Radiación Ultravioleta (UV)")
day_of_year_uv = st.slider("Día del Año (Radiación UV)", 1, 365, 172)
df_uv = generate_radiation_data(latitude, day_of_year_uv, "Radiación UV")
fig_uv = px.line(
    df_uv,
    x="Hora del Día",
    y="Radiación (W/m²)",
    title="Radiación UV durante el Día",
    labels={"Hora del Día": "Hora del Día", "Radiación (W/m²)": "Radiación UV (W/m²)"},
)
st.plotly_chart(fig_uv)
