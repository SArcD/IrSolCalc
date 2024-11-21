import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Funciones necesarias
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
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    elevation = math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

    cos_azimuth = (math.sin(math.radians(declination)) - 
                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))
    azimuth = math.degrees(math.acos(cos_azimuth)) if elevation > 0 else 0

    if hour_angle > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

def generate_solar_path(latitude, fixed_hour):
    """Genera los datos para azimut vs elevación solar."""
    days_of_year = np.arange(1, 366)
    elevations = []
    azimuths = []
    days = []

    for day in days_of_year:
        declination = calculate_declination(day)
        eot = calculate_equation_of_time(day)
        hour_angle = calculate_hour_angle(fixed_hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation > 0:  # Ignorar valores negativos de elevación
            elevations.append(elevation)
            azimuths.append(azimuth)
            days.append(day)

    return pd.DataFrame({"Día del Año": days, "Azimut (°)": azimuths, "Elevación Solar (°)": elevations})

# Configuración de Streamlit
st.title("Calculadora de Radiación Solar y Posición del Sol")

# Primera sección: Gráfica de Azimut vs Elevación Solar
st.subheader("Gráfica de Azimut vs Elevación Solar")
latitude = st.slider("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)
fixed_hour = st.slider("Hora Fija (24h)", 0.0, 24.0, 12.0)

df = generate_solar_path(latitude, fixed_hour)

st.write(f"**Azimut y Elevación Solar** para Latitud {latitude}° y Hora Fija {fixed_hour}:00")
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


################################################################3

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Funciones necesarias
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
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    elevation = math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

    cos_azimuth = (math.sin(math.radians(declination)) - 
                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))
    azimuth = math.degrees(math.acos(cos_azimuth)) if elevation > 0 else 0

    if hour_angle > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

def generate_solar_path(latitude, fixed_hour):
    """Genera los datos para azimut vs elevación solar."""
    days_of_year = np.arange(1, 366)
    elevations = []
    azimuths = []
    days = []

    for day in days_of_year:
        declination = calculate_declination(day)
        eot = calculate_equation_of_time(day)
        hour_angle = calculate_hour_angle(fixed_hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation > 0:  # Ignorar valores negativos de elevación
            elevations.append(elevation)
            azimuths.append(azimuth)
            days.append(day)

    return pd.DataFrame({"Día del Año": days, "Azimut (°)": azimuths, "Elevación Solar (°)": elevations})

# Configuración de Streamlit
st.title("Calculadora de Radiación Solar y Posición del Sol en Coordenadas Esféricas")

# Inputs
latitude = st.slider("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)
fixed_hour = st.slider("Hora Fija (24h)", 0.0, 24.0, 12.0)

# Generar datos de trayectoria solar
df = generate_solar_path(latitude, fixed_hour)

# Convertir a coordenadas esféricas (radio unitario)
solar_positions = [
    (
        math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim)),
        math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim)),
        math.cos(math.radians(90 - elev))
    )
    for elev, azim in zip(df["Elevación Solar (°)"], df["Azimut (°)"])
]

solar_x, solar_y, solar_z = zip(*solar_positions)

# Crear la esfera como referencia
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi / 2, 100)  # Media esfera
x = np.outer(np.sin(phi), np.cos(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.cos(phi), np.ones_like(theta))

# Crear gráfica 3D interactiva
fig = go.Figure()

# Agregar la media esfera
fig.add_trace(go.Surface(
    x=x, y=y, z=z,
    colorscale='Blues',
    opacity=0.3,
    name="Media Esfera Celeste",
    showscale=False
))

# Agregar la trayectoria solar
fig.add_trace(go.Scatter3d(
    x=solar_x,
    y=solar_y,
    z=solar_z,
    mode='markers+lines',
    marker=dict(size=6, color=df["Día del Año"], colorscale="Viridis", colorbar=dict(title="Día del Año")),
    hovertemplate=(
        "Día del Año: %{customdata[0]}<br>" +
        "Azimut: %{customdata[1]:.2f}°<br>" +
        "Elevación: %{customdata[2]:.2f}°"
    ),
    customdata=np.stack((df["Día del Año"], df["Azimut (°)"], df["Elevación Solar (°)"]), axis=-1),
    name="Posición Solar"
))

# Agregar plano del horizonte
x_horiz = np.linspace(-1, 1, 100)
y_horiz = np.linspace(-1, 1, 100)
x_horiz, y_horiz = np.meshgrid(x_horiz, y_horiz)
z_horiz = np.zeros_like(x_horiz)

fig.add_trace(go.Surface(
    x=x_horiz, y=y_horiz, z=z_horiz,
    colorscale='Greens',
    opacity=0.5,
    name="Plano del Horizonte",
    showscale=False
))

# Configurar vista
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z (Elevación)"
    ),
    title="Posición Solar en Coordenadas Esféricas",
    height=700,
    width=900
)

st.plotly_chart(fig)



#import math
#import numpy as np
#import pandas as pd
#import plotly.express as px
#import streamlit as st

# Funciones necesarias (ya definidas previamente)
#def calculate_declination(day_of_year):
#    """Calcula la declinación solar en función del día del año."""
#    return 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

#def calculate_equation_of_time(day_of_year):
#    """Calcula la ecuación del tiempo en minutos."""
#    B = math.radians((360 / 365) * (day_of_year - 81))
#    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

#def calculate_hour_angle(hour, equation_of_time):
#    """Corrige el ángulo horario por la ecuación del tiempo."""
#    solar_time = hour + (equation_of_time / 60)
#    return 15 * (solar_time - 12)

#def calculate_solar_position(latitude, declination, hour_angle):
#    """Calcula la elevación y azimut solar en grados."""
#    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
#                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
#    elevation = math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0
#
#    cos_azimuth = (math.sin(math.radians(declination)) - 
#                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
#                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))
#    azimuth = math.degrees(math.acos(cos_azimuth)) if elevation > 0 else 0

#    if hour_angle > 0:
#        azimuth = 360 - azimuth

#    return elevation, azimuth

#def generate_daily_solar_position(latitude, day_of_year):
#    """Genera los datos de posición solar para todas las horas del día."""
#    hours = np.arange(0, 24, 0.5)  # Horas del día en pasos de 0.5
#    elevations = []
#    azimuths = []
#    hours_list = []

#    declination = calculate_declination(day_of_year)
#    eot = calculate_equation_of_time(day_of_year)

#    for hour in hours:
#        hour_angle = calculate_hour_angle(hour, eot)
#        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

 #       if elevation > 0:  # Ignorar valores negativos (noche)
 #           elevations.append(elevation)
 #           azimuths.append(azimuth)
 #           hours_list.append(hour)

 #   return pd.DataFrame({
 #       "Hora del Día": hours_list,
 #       "Elevación Solar (°)": elevations,
 #       "Azimut Solar (°)": azimuths
 #   })

# Nueva sección: Posición Solar a lo Largo del Día
#st.subheader("Gráfica de Posición Solar a lo Largo del Día")
#day_of_year_position = st.slider("Día del Año (Posición Solar)", 1, 365, 172)
#latitude_position = st.slider("Latitud (°) para Posición Solar", -90.0, 90.0, 19.43, step=0.1)

#df_position = generate_daily_solar_position(latitude_position, day_of_year_position)

# Mostrar la gráfica interactiva
#st.write(f"**Posición Solar** para Latitud {latitude_position}° y Día del Año {day_of_year_position}")
#fig_position = px.line(
#    df_position,
#    x="Hora del Día",
#    y=["Elevación Solar (°)", "Azimut Solar (°)"],
#    title="Posición Solar a lo Largo del Día",
#    labels={"value": "Ángulo (°)", "Hora del Día": "Hora Local"},
#    markers=True
#)
#fig_position.update_layout(
#    xaxis_title="Hora del Día",
#    yaxis_title="Ángulo Solar (°)",
#    legend_title="Tipo de Ángulo",
#    height=600,
#    width=900
#)
#st.plotly_chart(fig_position)

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

## Funciones necesarias
#def calculate_declination(day_of_year):
#    """Calcula la declinación solar en función del día del año."""
#    return 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

#def calculate_equation_of_time(day_of_year):
#    """Calcula la ecuación del tiempo en minutos."""
#    B = math.radians((360 / 365) * (day_of_year - 81))
#    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

#def calculate_hour_angle(hour, equation_of_time):
#    """Corrige el ángulo horario por la ecuación del tiempo."""
#    solar_time = hour + (equation_of_time / 60)
#    return 15 * (solar_time - 12)

#def calculate_solar_position(latitude, declination, hour_angle):
#    """Calcula la elevación solar (altitud) en grados."""
#    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
#                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
#    elevation = math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0
#
#    cos_azimuth = (math.sin(math.radians(declination)) - 
#                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
#                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))
#    azimuth = math.degrees(math.acos(cos_azimuth)) if elevation > 0 else 0

#    if hour_angle > 0:
#        azimuth = 360 - azimuth

#    return elevation, azimuth

#def generate_daily_solar_position(latitude, day_of_year):
#    """Genera los datos de posición solar para todas las horas del día."""
#    hours = np.arange(0, 24, 0.5)  # Horas del día en pasos de 0.5
#    elevations = []
#    azimuths = []
#    hours_list = []

#    declination = calculate_declination(day_of_year)
#    eot = calculate_equation_of_time(day_of_year)

#    for hour in hours:
#        hour_angle = calculate_hour_angle(hour, eot)
#        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

#        if elevation > 0:  # Ignorar valores negativos (noche)
#            elevations.append(elevation)
#            azimuths.append(azimuth)
#            hours_list.append(hour)

#    return pd.DataFrame({
#        "Hora del Día": hours_list,
#        "Elevación Solar (°)": elevations,
#        "Azimut Solar (°)": azimuths
#    })

# Configuración de Streamlit
#st.title("Posición Solar en 3D a lo Largo del Día")

# Inputs del usuario
#latitude = st.slider("Latitud (en grados)", -90.0, 90.0, 19.43, step=0.1)
#day_of_year = st.slider("Día en el Año", 1, 365, 172)
#view_angle = st.slider("Ángulo de visión (°) (0° = Norte)", 0, 360, 0)

# Generar datos de posición solar
#df_position = generate_daily_solar_position(latitude, day_of_year)

# Gráfica 3D interactiva
#st.write(f"**Gráfica 3D de la Posición Solar** para Latitud {latitude}° y Día del Año {day_of_year}")
#fig = go.Figure()

#fig.add_trace(go.Scatter3d(
#    x=df_position["Azimut Solar (°)"],
#    y=df_position["Hora del Día"],
#    z=df_position["Elevación Solar (°)"],
#    mode='markers+lines',
#    marker=dict(size=4, color=df_position["Hora del Día"], colorscale='Viridis', colorbar=dict(title="Hora del Día")),
#    line=dict(color='blue'),
#    name="Posición Solar"
#))

#fig.update_layout(
#    scene=dict(
#        xaxis_title="Azimut Solar (°)",
#        yaxis_title="Hora del Día",
#        zaxis_title="Elevación Solar (°)",
#        camera=dict(
#            eye=dict(x=2 * math.cos(math.radians(view_angle)), 
#                     y=2 * math.sin(math.radians(view_angle)), 
#                     z=1.5)  # Ajuste de altura del observador
#        )
#    ),
#    height=700,
#    width=900,
#    title="Posición Solar en 3D a lo Largo del Día"
#)

#st.plotly_chart(fig)


import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Funciones necesarias
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
    """Calcula la elevación solar (altitud) y azimut en grados."""
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    elevation = math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

    cos_azimuth = (math.sin(math.radians(declination)) - 
                   math.sin(math.radians(latitude)) * math.sin(math.radians(elevation))) / (
                   math.cos(math.radians(latitude)) * math.cos(math.radians(elevation)))
    azimuth = math.degrees(math.acos(cos_azimuth)) if elevation > 0 else 0

    if hour_angle > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

def generate_daily_solar_position(latitude, day_of_year):
    """Genera los datos de posición solar para todas las horas del día."""
    hours = np.arange(0, 24, 0.5)  # Horas del día en pasos de 0.5
    elevations = []
    azimuths = []
    hours_list = []

    declination = calculate_declination(day_of_year)
    eot = calculate_equation_of_time(day_of_year)

    for hour in hours:
        hour_angle = calculate_hour_angle(hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation > 0:  # Ignorar valores negativos (noche)
            elevations.append(elevation)
            azimuths.append(azimuth)
            hours_list.append(hour)

    return pd.DataFrame({
        "Hora del Día": hours_list,
        "Elevación Solar (°)": elevations,
        "Azimut Solar (°)": azimuths
    })

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Funciones necesarias
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
    """Calcula la elevación solar (altitud) y azimut en grados."""
    sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
    if sin_altitude <= 0:  # Si el Sol está por debajo del horizonte
        return None, None

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
    hours = np.arange(0, 24, 0.5)  # Horas del día en pasos de 0.5
    elevations, azimuths, hours_list = [], [], []

    declination = calculate_declination(day_of_year)
    eot = calculate_equation_of_time(day_of_year)

    for hour in hours:
        hour_angle = calculate_hour_angle(hour, eot)
        elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

        if elevation is not None and azimuth is not None:
            elevations.append(elevation)
            azimuths.append(azimuth)
            hours_list.append(hour)

    return pd.DataFrame({
        "Hora del Día": hours_list,
        "Elevación Solar (°)": elevations,
        "Azimut Solar (°)": azimuths
    })

# Configuración de Streamlit
st.title("Vista del Observador: Posición Solar basada en la Hora")

# Inputs del usuario
latitude = st.slider("Latitud", -90.0, 90.0, 19.43, step=0.1)
day_of_year = st.slider("Día", 1, 365, 172)

# Generar datos de posición solar
df_position = generate_daily_solar_position(latitude, day_of_year)

# Deslizador para seleccionar la hora
selected_hour = st.slider("Selecciona la hora del día:", min_value=0.0, max_value=24.0, step=0.5, value=12.0)

# Encontrar los datos de la posición solar en la hora seleccionada
selected_row = df_position[df_position["Hora del Día"] == selected_hour]
if not selected_row.empty:
    elev = selected_row["Elevación Solar (°)"].values[0]
    azim = selected_row["Azimut Solar (°)"].values[0]
else:
    elev = azim = 0

# Transformar coordenadas esféricas a cartesianas para la trayectoria solar
solar_positions = [
    (
        math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim)),
        math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim)),
        math.cos(math.radians(90 - elev))
    )
    for elev, azim in zip(df_position["Elevación Solar (°)"], df_position["Azimut Solar (°)"])
]

solar_x, solar_y, solar_z = zip(*solar_positions)

# Coordenadas de la flecha para la hora seleccionada
arrow_x = math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim))
arrow_y = math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim))
arrow_z = math.cos(math.radians(90 - elev))

# Crear la media esfera
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi / 2, 100)  # Solo la mitad superior
x = np.outer(np.sin(phi), np.cos(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.cos(phi), np.ones_like(theta))

# Gráfica 3D interactiva
fig = go.Figure()

# Agregar la media esfera
fig.add_trace(go.Surface(
    x=x, y=y, z=z, 
    colorscale='Blues',
    opacity=0.3,
    name="Media Esfera Celeste",
    showscale=False
))

# Agregar plano del horizonte
x_horiz = np.linspace(-1, 1, 100)
y_horiz = np.linspace(-1, 1, 100)
x_horiz, y_horiz = np.meshgrid(x_horiz, y_horiz)
z_horiz = np.zeros_like(x_horiz)

fig.add_trace(go.Surface(
    x=x_horiz, y=y_horiz, z=z_horiz,
    colorscale='Greens',
    opacity=0.5,
    name="Plano del Horizonte",
    showscale=False
))

# Agregar posiciones solares con hover limitado
fig.add_trace(go.Scatter3d(
    x=solar_x,
    y=solar_y,
    z=solar_z,
    mode='markers+lines',
    marker=dict(size=6, color="orange"),
    name="Posición Solar"
))

# Agregar flecha para la hora seleccionada
fig.add_trace(go.Scatter3d(
    x=[0, arrow_x],
    y=[0, arrow_y],
    z=[0, arrow_z],
    mode="lines+text",
    line=dict(color="blue", width=5),
    text=f"Hora: {selected_hour}h<br>Azimut: {azim:.2f}°<br>Elevación: {elev:.2f}°",
    textposition="top center",
    name="Posición Solar Actual"
))

fig.update_layout(
    scene=dict(
        xaxis_title="X (Azimut)",
        yaxis_title="Y",
        zaxis_title="Z (Elevación)"
    ),
    height=700,
    width=900,
    title="Vista del Observador: Movimiento del Sol"
)

st.plotly_chart(fig)

# Segunda sección: Cálculo de radiación solar
st.subheader("Cálculo de Radiación Solar")
day_of_year = st.slider("Día del Año", 1, 365, 172)
local_hour = st.slider("Hora Local (24h)", 0.0, 24.0, 12.0)
transmission_coefficient = st.slider("Coeficiente de Transmisión Atmosférica", 0.0, 1.0, 0.75)

def calculate_solar_power(latitude, day_of_year, local_hour, transmission_coefficient):
    S0 = 1361  # Constante solar (W/m²)
    declination = calculate_declination(day_of_year)
    solar_hour = local_hour - 12
    hour_angle = 15 * solar_hour

    sin_alpha = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                 math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))

    if sin_alpha <= 0:
        return 0

    return S0 * transmission_coefficient * sin_alpha

power = calculate_solar_power(latitude, day_of_year, local_hour, transmission_coefficient)
st.write(f"La potencia de radiación solar recibida es de aproximadamente **{power:.2f} W/m²**.")



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
