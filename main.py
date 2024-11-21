import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# Configuración de Streamlit
st.title("Posición Solar y Radiación Solar en Coordenadas Esféricas")

# Barra lateral para los inputs
st.sidebar.header("Parámetros de Entrada")
latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)
day_of_year = st.sidebar.slider("Día del Año", 1, 365, 172)
selected_hour = st.sidebar.slider("Hora del Día (24h)", 0.0, 24.0, 12.0, step=0.5)

# Generar datos de posición solar
df_position = generate_daily_solar_position(latitude, day_of_year)

# Seleccionar posición solar para la hora elegida
selected_row = df_position[df_position["Hora del Día"] == selected_hour]
if not selected_row.empty:
    elev = selected_row["Elevación Solar (°)"].values[0]
    azim = selected_row["Azimut Solar (°)"].values[0]
else:
    elev = azim = 0

# Transformar a coordenadas esféricas
solar_positions = [
    (
        math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim)),
        math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim)),
        math.cos(math.radians(90 - elev))
    )
    for elev, azim in zip(df_position["Elevación Solar (°)"], df_position["Azimut Solar (°)"])
]

solar_x, solar_y, solar_z = zip(*solar_positions)

# Coordenadas para la flecha
arrow_x = math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim))
arrow_y = math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim))
arrow_z = math.cos(math.radians(90 - elev))

# Crear la media esfera
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi / 2, 100)
x = np.outer(np.sin(phi), np.cos(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.cos(phi), np.ones_like(theta))

# Crear gráfica 3D interactiva
fig_position = go.Figure()

# Media esfera
fig_position.add_trace(go.Surface(
    x=x, y=y, z=z,
    colorscale='Blues',
    opacity=0.3,
    showscale=False,
    name="Media Esfera Celeste"
))

# Trayectoria solar
fig_position.add_trace(go.Scatter3d(
    x=solar_x,
    y=solar_y,
    z=solar_z,
    mode='markers+lines',
    marker=dict(size=6, color="orange"),
    name="Trayectoria Solar"
))

# Flecha para la hora seleccionada
fig_position.add_trace(go.Scatter3d(
    x=[0, arrow_x],
    y=[0, arrow_y],
    z=[0, arrow_z],
    mode="lines+text",
    line=dict(color="blue", width=5),
    text=f"Hora: {selected_hour}h<br>Azimut: {azim:.2f}°<br>Elevación: {elev:.2f}°",
    textposition="top center",
    name="Posición Solar Actual"
))

# Plano del horizonte
x_horiz = np.linspace(-1, 1, 100)
y_horiz = np.linspace(-1, 1, 100)
x_horiz, y_horiz = np.meshgrid(x_horiz, y_horiz)
z_horiz = np.zeros_like(x_horiz)

fig_position.add_trace(go.Surface(
    x=x_horiz, y=y_horiz, z=z_horiz,
    colorscale='Greens',
    opacity=0.5,
    showscale=False,
    name="Plano del Horizonte"
))

fig_position.update_layout(
    scene=dict(
        xaxis_title="X (Azimut)",
        yaxis_title="Y",
        zaxis_title="Z (Elevación)"
    ),
    height=700,
    width=900,
    title="Vista del Observador: Movimiento del Sol"
)

# Pestañas en Streamlit
tab1, tab2 = st.tabs(["Posición Solar", "Cálculo de Radiación"])

with tab1:
    st.plotly_chart(fig_position, use_container_width=True)

with tab2:
    # Cálculo de radiación solar
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
