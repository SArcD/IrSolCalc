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

def generate_radiation_data(latitude, fixed_hour, radiation_percentage):
    """Genera los datos de radiación para cada día del año."""
    days_of_year = np.arange(1, 366)  # Días del año
    radiations = []
    altitudes = []

    for day in days_of_year:
        declination = calculate_declination(day)
        eot = calculate_equation_of_time(day)  # Ecuación del tiempo
        hour_angle = calculate_hour_angle(fixed_hour, eot)
        altitude = calculate_solar_position(latitude, declination, hour_angle)
        total_radiation = calculate_radiation(altitude)
        radiation = total_radiation * radiation_percentage

        altitudes.append(altitude)
        radiations.append(radiation)

    return pd.DataFrame({"Día del Año": days_of_year, "Altitud Solar (°)": altitudes, "Radiación (W/m²)": radiations})

# Configuración de Streamlit
st.title("Variación de Radiación Solar Incidente")
#st.sidebar.header("Parámetros de Entrada")

# Entrada del usuario
#latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43)
#fixed_hour = st.sidebar.slider("Hora Fija (24h)", 0.0, 24.0, 12.0)

# Selección del tipo de radiación
radiation_type = st.sidebar.selectbox(
    "Selecciona el tipo de radiación:",
    ["Total", "Ultravioleta (UV)", "Rayos X", "Microondas", "Radio"]
)

# Porcentajes de la radiación
radiation_percentages = {
    "Total": 1.0,           # 100% de la radiación total
    "Ultravioleta (UV)": 0.05,  # 5% de la radiación total
    "Rayos X": 0.00001,     # 0.001% (aproximado)
    "Microondas": 0.0001,   # 0.01% (aproximado)
    "Radio": 0.0001         # 0.01% (aproximado)
}

# Obtener el porcentaje correspondiente
radiation_percentage = radiation_percentages[radiation_type]

# Generar datos
df = generate_radiation_data(latitude, fixed_hour, radiation_percentage)

# Mostrar los datos generados
st.write(f"**Datos de Radiación Solar** para Latitud {latitude}° y Hora Fija {fixed_hour}:00 - {radiation_type}")
st.dataframe(df)

# Gráfica interactiva
st.write(f"**Gráfica de Variación de Radiación Solar** ({radiation_type})")
fig = px.line(
    df,
    x="Día del Año",
    y="Radiación (W/m²)",
    title=f"Variación de Radiación Solar para Latitud {latitude}° - Hora Fija: {fixed_hour}:00 ({radiation_type})",
    labels={"Día del Año": "Día del Año", "Radiación (W/m²)": "Radiación (W/m²)"},
)
fig.update_layout(
    xaxis_title="Día del Año",
    yaxis_title="Radiación Solar (W/m²)",
    height=600,
    width=900
)
st.plotly_chart(fig)
###############################################################


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
st.title("Variación de Radiación Solar a lo Largo del Día")
st.sidebar.header("Parámetros de Entrada")

# Entrada del usuario
#latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43)
day_of_year = st.sidebar.slider("Día del Año", 1, 365, 172)
radiation_type = st.sidebar.selectbox("Selecciona el Tipo de Radiación", ["Radiación Total", "Radiación UV"])

# Generar datos
df = generate_radiation_data(latitude, day_of_year, radiation_type)

# Mostrar los datos generados
st.write(f"**Datos de Radiación Solar ({radiation_type})** para Latitud {latitude}° y Día del Año {day_of_year}")
st.dataframe(df)

# Gráfica interactiva
st.write(f"**Gráfica de Variación de {radiation_type}**")
fig = px.line(
    df,
    x="Hora del Día",
    y="Radiación (W/m²)",
    title=f"Variación de {radiation_type} para Latitud {latitude}° - Día del Año {day_of_year}",
    labels={"Hora del Día": "Hora del Día", "Radiación (W/m²)": f"{radiation_type} (W/m²)"},
)
fig.update_layout(
    xaxis_title="Hora del Día",
    yaxis_title=f"{radiation_type} (W/m²)",
    height=600,
    width=900
)
st.plotly_chart(fig)
