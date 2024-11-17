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
    """Calcula la elevación solar (altitud) en grados."""
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    return math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

def calculate_radiation(altitude):
    """Calcula la radiación solar incidente en W/m²."""
    # Constante solar (W/m²)
    S0 = 1361
    # Transmisión atmosférica promedio
    T_a = 0.75
    # Radiación incidente ajustada por el ángulo de incidencia
    return S0 * T_a * math.sin(math.radians(altitude)) if altitude > 0 else 0

def generate_radiation_data(latitude, fixed_hour):
    """Genera los datos de radiación para cada día del año."""
    days_of_year = np.arange(1, 366)  # Días del año
    radiations = []
    altitudes = []

    for day in days_of_year:
        declination = calculate_declination(day)
        eot = calculate_equation_of_time(day)  # Ecuación del tiempo
        hour_angle = calculate_hour_angle(fixed_hour, eot)
        altitude = calculate_solar_position(latitude, declination, hour_angle)
        radiation = calculate_radiation(altitude)

        altitudes.append(altitude)
        radiations.append(radiation)

    return pd.DataFrame({"Día del Año": days_of_year, "Altitud Solar (°)": altitudes, "Radiación (W/m²)": radiations})

# Configuración de Streamlit
st.title("Variación de Radiación Solar Incidente")
st.sidebar.header("Parámetros de Entrada")

# Entrada del usuario
latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43)
fixed_hour = st.sidebar.slider("Hora Fija (24h)", 0.0, 24.0, 12.0)

# Generar datos
df = generate_radiation_data(latitude, fixed_hour)

# Mostrar los datos generados
st.write(f"**Datos de Radiación Solar** para Latitud {latitude}° y Hora Fija {fixed_hour}:00")
st.dataframe(df)

# Gráfica interactiva
st.write(f"**Gráfica de Variación de Radiación Solar**")
fig = px.line(
    df,
    x="Día del Año",
    y="Radiación (W/m²)",
    title=f"Variación de Radiación Solar para Latitud {latitude}° - Hora Fija: {fixed_hour}:00",
    labels={"Día del Año": "Día del Año", "Radiación (W/m²)": "Radiación (W/m²)"},
)
fig.update_layout(
    xaxis_title="Día del Año",
    yaxis_title="Radiación Solar (W/m²)",
    height=600,
    width=900
)
st.plotly_chart(fig)
