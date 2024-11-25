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


# Pestañas en Streamlit
tab1, tab2 = st.tabs(["Posición Solar", "Cálculo de Radiación"])

with tab1:

    # Configuración de Streamlit
    st.title("Vista del Observador: Posición Solar y Radiación Solar")

    # Barra lateral para los inputs
    st.sidebar.header("Parámetros de Entrada")
    latitude = st.sidebar.slider("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)
    latitude=-latitude
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

    # Gráfica 3D
    fig = go.Figure()

    # Media esfera
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        opacity=0.3,
        showscale=False,
        name="Media Esfera Celeste"
    ))

    # Trayectoria solar
    fig.add_trace(go.Scatter3d(
        x=solar_x,
        y=solar_y,
        z=solar_z,
        mode='markers+lines',
        marker=dict(size=6, color="orange"),
        name="Trayectoria Solar"
    ))

    # Flecha para la hora seleccionada
    #fig.add_trace(go.Scatter3d(
    #    x=[0, arrow_x],
    #    y=[0, arrow_y],
    #    z=[0, arrow_z],
    #    mode="lines+text",
    #    line=dict(color="blue", width=5),
    #    text=f"Hora: {selected_hour}h<br>Azimut: {azim:.2f}°<br>Elevación: {elev:.2f}°",
    #    textposition="top center",
    #    name="Posición Solar Actual"
    #))


    # Flecha para la hora seleccionada
    fig.add_trace(go.Scatter3d(
        x=[0, arrow_x],  # Coordenadas de la flecha
        y=[0, arrow_y],
        z=[0, arrow_z],
        mode="lines+text",
        line=dict(color="blue", width=5),
        text=[None, f"Hora: {selected_hour}h<br>Azimut: {azim:.2f}°<br>Elevación: {elev:.2f}°"],  # Solo texto en el extremo
        textposition="top center",  # Posición del texto
        name="Posición Solar Actual"
    ))
    
    
    # Plano del horizonte
    x_horiz = np.linspace(-1, 1, 100)
    y_horiz = np.linspace(-1, 1, 100)
    x_horiz, y_horiz = np.meshgrid(x_horiz, y_horiz)
    z_horiz = np.zeros_like(x_horiz)

    fig.add_trace(go.Surface(
        x=x_horiz, y=y_horiz, z=z_horiz,
        colorscale='Greens',
        opacity=0.5,
        showscale=False,
        name="Plano del Horizonte"
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


    directions = {
        "Sur": (1, 0, 0),   # Eje positivo en Y
        "Este": (0, 1, 0),    # Eje positivo en X
        "Norte": (-1, 0, 0),    # Eje negativo en Y
        "Oeste": (0, -1, 0)   # Eje negativo en X
    }

    for name, coord in directions.items():
        fig.add_trace(go.Scatter3d(
            x=[0, coord[0]],
            y=[0, coord[1]],
            z=[0, coord[2]],
            mode="lines+text",
            text=[None, name],
            textposition="top center",
            line=dict(color="red", width=4),
            name=name
        ))

    st.plotly_chart(fig)
##########################################################################33




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

    def generate_solar_path(latitude, selected_hour):
        """Genera los datos para azimut y elevación solar."""
        days_of_year = np.arange(1, 366)
        elevations, azimuths, days = [], [], []

        for day in days_of_year:
            declination = calculate_declination(day)
            eot = calculate_equation_of_time(day)
            hour_angle = calculate_hour_angle(selected_hour, eot)
            elevation, azimuth = calculate_solar_position(latitude, declination, hour_angle)

            if elevation is not None:
                elevations.append(elevation)
                azimuths.append(azimuth)
                days.append(day)

        return pd.DataFrame({"Día del Año": days, "Azimut (°)": azimuths, "Elevación Solar (°)": elevations})

    # Configuración de Streamlit
    st.title("Calculadora de Radiación Solar y Posición del Sol en Coordenadas Esféricas")

## Inputs del usuario
#latitude = st.slider("Latitud (°)", -90.0, 90.0, 19.43, step=0.1)
#_hour = st.slider("Hora Fija (24h)", 0.0, 24.0, 12.0)

    # Generar datos de trayectoria solar
    df = generate_solar_path(latitude, selected_hour)

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

    # Obtener elevación y azimut de la flecha
    elev = df["Elevación Solar (°)"].iloc[-1]
    azim = df["Azimut (°)"].iloc[-1]
    arrow_x = math.sin(math.radians(90 - elev)) * math.cos(math.radians(azim))
    arrow_y = math.sin(math.radians(90 - elev)) * math.sin(math.radians(azim))
    arrow_z = math.cos(math.radians(90 - elev))

    # Crear la esfera como referencia
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi / 2, 100)  # Media esfera
    x = np.outer(np.sin(phi), np.cos(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.cos(phi), np.ones_like(theta))

    # Crear gráfica 3D interactiva
    fig = go.Figure()

    # Media esfera
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        opacity=0.3,
        name="Media Esfera Celeste",
        showscale=False
    ))

    # Trayectoria solar
    fig.add_trace(go.Scatter3d(
        x=solar_x,
        y=solar_y,
        z=solar_z,
        mode='markers+lines',
        marker=dict(size=6, color=df["Día del Año"], colorscale="Viridis", colorbar=dict(title="Día del Año"), showscale=False),
        hovertemplate=(
            "Día del Año: %{customdata[0]}<br>" +
            "Azimut: %{customdata[1]:.2f}°<br>" +
            "Elevación: %{customdata[2]:.2f}°"
        ),
        customdata=np.stack((df["Día del Año"], df["Azimut (°)"], df["Elevación Solar (°)"]), axis=-1),
        name="Posición Solar"
    ))


    # Flecha para la hora seleccionada
    fig.add_trace(go.Scatter3d(
        x=[0, arrow_x],  # Coordenadas de la flecha
        y=[0, arrow_y],
        z=[0, arrow_z],
        mode="lines+text",
        line=dict(color="blue", width=5),
        text=[None, f"Hora: {selected_hour}h<br>Azimut: {azim:.2f}°<br>Elevación: {elev:.2f}°"],  # Solo texto en el extremo
        textposition="top center",  # Posición del texto
        name="Posición Solar Actual"
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


    # Agregar flechas y etiquetas de los puntos cardinales
    #directions = {
    #    "Norte": (0, 0.5, 0),
    #    "Este": (0.5, 0, 0),
    #    "Sur": (0, -0.5, 0),
    #    "Oeste": (-0.5, 0, 0)
    #}

    #directions = {
    #"Norte": (0, 1, 0),   # Norte en el eje positivo Y
    #"Este": (1, 0, 0),    # Este en el eje positivo X
    #"Sur": (0, -1, 0),    # Sur en el eje negativo Y
    #"Oeste": (-1, 0, 0)   # Oeste en el eje negativo X
    #}


    directions = {
        "Este": (0, 1, 0),
        "Sur": (1, 0, 0),
        "Oeste": (0, -1, 0),
        "Norte": (-1, 0, 0)
    }



    
    for name, coord in directions.items():
        fig.add_trace(go.Scatter3d(
            x=[0, coord[0]],
            y=[0, coord[1]],
            z=[0, coord[2]],
            mode="lines+text",
            text=[None, name],
            textposition="top center",
            line=dict(color="red", width=4),
            name=name
        ))

    

    st.plotly_chart(fig)


    ############################################################################

with tab2:


    # Sección de Radiación Solar
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
    st.write(f"La radiación solar total recibida es **{radiation_power:.2f} W/m²- Día del Año {day_of_year}- Hora {selected_hour}")
    st.write(f"La radiación solar UV recibida es **{0.05*radiation_power:.2f} W/m²- Día del Año {day_of_year}- Hora {selected_hour}")

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
        """Calcula la elevación solar (altitud) en grados."""
        sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                        math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
        return math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

    def calculate_radiation(altitude):
        """Calcula la radiación solar incidente en W/m²."""
        S0 = 1361  # Constante solar (W/m²)
        T_a = 0.75  # Transmisión atmosférica promedio
        return S0 * T_a * math.sin(math.radians(altitude)) if altitude > 0 else 0

    def calculate_uv_radiation(total_radiation):
        """Calcula la fracción de radiación solar correspondiente a la luz UV."""
        uv_fraction = 0.05  # 5% de la radiación total
        return total_radiation * uv_fraction

    def generate_radiation_data(latitude, selected_hour, radiation_type="Total"):
        """Genera los datos de radiación para cada día del año."""
        days_of_year = np.arange(1, 366)  # Días del año
        radiations = []
        altitudes = []

        for day in days_of_year:
            declination = calculate_declination(day)
            eot = calculate_equation_of_time(day)  # Ecuación del tiempo
            hour_angle = calculate_hour_angle(selected_hour, eot)
            altitude = calculate_solar_position(latitude, declination, hour_angle)
            total_radiation = calculate_radiation(altitude)

            if radiation_type == "Total":
                radiation = total_radiation
            elif radiation_type == "UV":
                radiation = calculate_uv_radiation(total_radiation)
            else:
                radiation = 0  # Default case

            altitudes.append(altitude)
            radiations.append(radiation)

        return pd.DataFrame({"Día del Año": days_of_year, "Altitud Solar (°)": altitudes, "Radiación (W/m²)": radiations})

    # Configuración de Streamlit
    #st.title("Variación de Radiación Solar")
    #st.write("Explora cómo varía la radiación solar a lo largo del año según la latitud y la hora fija.")


    # Pestañas para elegir entre radiación total o UV
    tab1, tab2 = st.tabs(["Radiación Total", "Radiación UV"])

    with tab1:
        st.subheader("Radiación Solar Total")
        df_total = generate_radiation_data(latitude, selected_hour, radiation_type="Total")
        fig_total = px.line(
            df_total,
            x="Día del Año",
            y="Radiación (W/m²)",
            title=f"Variación de Radiación Solar Total para Latitud {latitude}° - Hora Fija: {selected_hour}:00",
            labels={"Día del Año": "Día del Año", "Radiación (W/m²)": "Radiación Total (W/m²)"},
        )
        fig_total.update_layout(
            xaxis_title="Día del Año",
            yaxis_title="Radiación Solar Total (W/m²)",
            height=600,
            width=900
        )
        st.plotly_chart(fig_total)

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
            """Calcula la elevación solar (altitud) en grados."""
            sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
            return math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

        def calculate_radiation(altitude):
            """Calcula la radiación solar total incidente en W/m²."""
            S0 = 1361  # Constante solar (W/m²)
            T_a = 0.75  # Transmisión atmosférica promedio
            return S0 * T_a * math.sin(math.radians(altitude)) if altitude > 0 else 0

        def generate_radiation_data(latitude, day_of_year):
            """Genera los datos de radiación total para cada hora del día."""
            hours_of_day = np.arange(0, 24, 0.5)  # Horas del día en intervalos de 0.5 horas
            radiations = []
            altitudes = []

            declination = calculate_declination(day_of_year)
            eot = calculate_equation_of_time(day_of_year)  # Ecuación del tiempo

            for hour in hours_of_day:
                hour_angle = calculate_hour_angle(hour, eot)
                altitude = calculate_solar_position(latitude, declination, hour_angle)
                total_radiation = calculate_radiation(altitude)

                altitudes.append(altitude)
                radiations.append(total_radiation)

            return pd.DataFrame({
                "Hora del Día": hours_of_day,
                "Altitud Solar (°)": altitudes,
                "Radiación Total (W/m²)": radiations
            })

        # Configuración de Streamlit
        #st.title("Variación de Radiación Total")
        st.write("Gráfica de la variación annual de la radiación solar total según la latitud y el día del año.")

    #day_of_year = st.sidebar.slider("Día del Año", 1, 365, 172)

    # Generar datos y gráfica
        df = generate_radiation_data(latitude, day_of_year)
        fig = px.line(
            df,
            x="Hora del Día",
            y="Radiación Total (W/m²)",
            title=f"Variación de Radiación Total para Latitud {latitude}° - Día del Año {day_of_year}",
            labels={"Hora del Día": "Hora del Día", "Radiación Total (W/m²)": "Radiación Total (W/m²)"},
        )
        fig.update_layout(
            xaxis_title="Hora del Día",
            yaxis_title="Radiación Total (W/m²)",
            height=600,
            width=900
        )

        # Mostrar la gráfica
        st.plotly_chart(fig)



    with tab2:

        def generate_radiation_data(latitude, selected_hour, radiation_type="UV"):
            """Genera los datos de radiación para cada día del año."""
            days_of_year = np.arange(1, 366)  # Días del año
            radiations = []
            altitudes = []

            for day in days_of_year:
                declination = calculate_declination(day)
                eot = calculate_equation_of_time(day)  # Ecuación del tiempo
                hour_angle = calculate_hour_angle(selected_hour, eot)
                altitude = calculate_solar_position(latitude, declination, hour_angle)
                total_radiation = calculate_radiation(altitude)

                if radiation_type == "Total":
                    radiation = total_radiation
                elif radiation_type == "UV":
                    radiation = calculate_uv_radiation(total_radiation)
                else:
                    radiation = 0  # Default case

                altitudes.append(altitude)
                radiations.append(radiation)

            return pd.DataFrame({"Día del Año": days_of_year, "Altitud Solar (°)": altitudes, "Radiación UV (W/m²)": radiations})





        st.subheader("Radiación Solar UV")
        df_total = generate_radiation_data(latitude, selected_hour, radiation_type="UV")
        fig_total = px.line(
            df_total,
            x="Día del Año",
            y="Radiación UV (W/m²)",
            title=f"Variación de Radiación Solar UV para Latitud {latitude}° - Hora Fija: {selected_hour}:00",
            labels={"Día del Año": "Día del Año", "Radiación UV (W/m²)": "Radiación UV (W/m²)"},
        )
        fig_total.update_layout(
            xaxis_title="Día del Año",
            yaxis_title="Radiación UV (W/m²)",
            height=600,
            width=900
        )
        st.plotly_chart(fig_total)

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
            """Calcula la elevación solar (altitud) en grados."""
            sin_altitude = (math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                    math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle)))
            return math.degrees(math.asin(sin_altitude)) if sin_altitude > 0 else 0

        def calculate_radiation(altitude):
            """Calcula la radiación solar UV incidente en W/m²."""
            S0 = 1361  # Constante solar (W/m²)
            T_a = 0.75  # Transmisión atmosférica promedio
            return S0 * T_a * math.sin(math.radians(altitude)) if altitude > 0 else 0

        def calculate_uv_radiation(total_radiation):
            """Calcula la fracción de radiación solar correspondiente a la luz UV."""
            uv_fraction = 0.05  # 5% de la radiación total
            return total_radiation * uv_fraction


        def calculate_daily_uv_radiation(latitude, day_of_year):
            """Calcula la radiación UV total para un día específico integrando numéricamente."""
            hours_of_day = np.linspace(0, 24, 100)  # Horas del día (más puntos para mayor precisión)
            declination = calculate_declination(day_of_year)
            eot = calculate_equation_of_time(day_of_year)  # Ecuación del tiempo

            uv_radiations = []
            for hour in hours_of_day:
                hour_angle = calculate_hour_angle(hour, eot)
                altitude = calculate_solar_position(latitude, declination, hour_angle)
                total_radiation = calculate_radiation(altitude)
                uv_radiation = calculate_uv_radiation(total_radiation)
                uv_radiations.append(uv_radiation)

            # Integrar numéricamente la radiación UV durante el día
            daily_uv = np.trapz(uv_radiations, hours_of_day)
            return daily_uv

        def calculate_annual_uv_radiation(latitude):
            """Calcula la radiación UV total para cada día del año y la acumula."""
            days_of_year = np.arange(1, 366)  # Días del año
            daily_uv_radiations = []

            for day in days_of_year:
                daily_uv = calculate_daily_uv_radiation(latitude, day)
                daily_uv_radiations.append(daily_uv)

            return pd.DataFrame({
                "Día del Año": days_of_year,
                "Radiación UV Diaria (Wh/m²)": daily_uv_radiations
            })


        
        def calculate_annual_uv_radiation(latitude):
            """Calcula la radiación UV total para cada día del año y la acumula."""
            days_of_year = np.arange(1, 366)  # Días del año
            daily_uv_radiations = []

            for day in days_of_year:
                daily_uv = calculate_daily_uv_radiation(latitude, day)
                daily_uv_radiations.append(daily_uv)

            return pd.DataFrame({
                "Día del Año": days_of_year,
                "Radiación UV Diaria (Wh/m²)": daily_uv_radiations
            })

        # Configuración de Streamlit
        #st.title("Variación de Radiación Total")
        st.write("Gráfica de la variación diaria de la radiación solar UV según la latitud y el día del año.")

    #day_of_year = st.sidebar.slider("Día del Año", 1, 365, 172)

        # Generar datos para la radiación UV
        def generate_uv_radiation_data(latitude, day_of_year):
            """Genera los datos de radiación UV para cada hora del día."""
            hours_of_day = np.arange(0, 24, 0.5)  # Horas del día en intervalos de 0.5 horas
            radiations = []
            uv_radiations = []
            altitudes = []

            declination = calculate_declination(day_of_year)
            eot = calculate_equation_of_time(day_of_year)  # Ecuación del tiempo

            for hour in hours_of_day:
                hour_angle = calculate_hour_angle(hour, eot)
                altitude = calculate_solar_position(latitude, declination, hour_angle)
                total_radiation = calculate_radiation(altitude)
                uv_radiation = calculate_uv_radiation(total_radiation)

                altitudes.append(altitude)
                radiations.append(total_radiation)
                uv_radiations.append(uv_radiation)

            return pd.DataFrame({
                "Hora del Día": hours_of_day,
                "Altitud Solar (°)": altitudes,
                "Radiación Total (W/m²)": radiations,
                "Radiación UV (W/m²)": uv_radiations
            })

        # Usar la columna de radiación UV en la gráfica
        df = generate_uv_radiation_data(latitude, day_of_year)
        fig = px.line(
            df,
            x="Hora del Día",
            y="Radiación UV (W/m²)",  # Aquí usamos la columna para radiación UV
            title=f"Variación de Radiación UV para Latitud {latitude}° - Día del Año {day_of_year}",
            labels={"Hora del Día": "Hora del Día", "Radiación UV (W/m²)": "Radiación UV (W/m²)"},
        )
        fig.update_layout(
            xaxis_title="Hora del Día",
            yaxis_title="Radiación UV (W/m²)",
            height=600,
            width=900
        )

        # Mostrar la gráfica
        st.plotly_chart(fig)

        #import folium
        #import streamlit as st
        #from streamlit_folium import st_folium

        # Título de la aplicación
        #st.title("Mapa Interactivo de México con Folium")
        #st.sidebar.header("Parámetros del Mapa")

        # Parámetros del mapa
        #latitude = st.sidebar.slider("Latitud inicial", -90.0, 90.0, 23.6345)
        #longitude = st.sidebar.slider("Longitud inicial", -180.0, 180.0, -102.5528)
        #zoom_level = st.sidebar.slider("Nivel de zoom", 1, 18, 5)

        # Crear el mapa centrado en la ubicación seleccionada
        #mapa_mexico = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)

        # Agregar un marcador en Ciudad de México
        #folium.Marker(
        #    location=[19.4326, -99.1332],
        #    popup="Ciudad de México",
        #    icon=folium.Icon(color="blue")
        #).add_to(mapa_mexico)

        # Agregar un marcador en Guadalajara
        #folium.Marker(
        #    location=[20.6597, -103.3496],
        #    popup="Guadalajara",
        #    icon=folium.Icon(color="green")
        #).add_to(mapa_mexico)

        # Agregar un marcador en Monterrey
        #folium.Marker(
        #    location=[25.6866, -100.3161],
        #    popup="Monterrey",
        #    icon=folium.Icon(color="red")
        #).add_to(mapa_mexico)

        # Mostrar el mapa en Streamlit
        #st_folium(mapa_mexico, width=800, height=500)


        import folium
        import streamlit as st
        from streamlit_folium import st_folium
        import math
        import geopandas as gpd
        import numpy as np

        import folium
        import geopandas as gpd
        import streamlit as st
        from streamlit_folium import st_folium
        import numpy as np

        import folium
        import geopandas as gpd
        import streamlit as st
        from streamlit_folium import st_folium
        import numpy as np

        # Parámetros de la fórmula    
        S0 = 1361  # Constante solar (W/m²)
        Ta = 0.75  # Transmisión atmosférica promedio    
        k = 0.12   # Incremento de radiación por km de altitud

        # Altitudes promedio estimadas por estado (en kilómetros)
        altitudes = {
            "Aguascalientes": 1.88, "Baja California": 0.58, "Baja California Sur": 0.40,
            "Campeche": 0.10, "Chiapas": 0.72, "Chihuahua": 1.49, "Ciudad de México": 2.24,
            "Coahuila": 1.12, "Colima": 0.33, "Durango": 1.88, "Guanajuato": 1.96,
            "Guerrero": 0.60, "Hidalgo": 1.90, "Jalisco": 1.56, "Estado de México": 2.57,
            "Michoacán": 1.75, "Morelos": 1.66, "Nayarit": 0.70, "Nuevo León": 1.57,
            "Oaxaca": 1.55, "Puebla": 2.13, "Querétaro": 1.82, "Quintana Roo": 0.10,
            "San Luis Potosí": 1.86, "Sinaloa": 0.38, "Sonora": 0.61, "Tabasco": 0.10,
            "Tamaulipas": 0.25, "Tlaxcala": 2.24, "Veracruz": 0.90, "Yucatán": 0.12, "Zacatecas": 2.19
        }

        # Función para calcular declinación solar
        def calculate_declination(day_of_year):
            return 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))

        # Función para calcular la radiación promedio anual
        def calculate_annual_radiation(latitude, altitude):
            total_radiation = 0
            for day in range(1, 366):
                declination = calculate_declination(day)
                sin_lat_decl = np.sin(np.radians(latitude)) * np.sin(np.radians(declination))
                cos_lat_decl = np.cos(np.radians(latitude)) * np.cos(np.radians(declination))
                daily_radiation = S0 * Ta * (sin_lat_decl + cos_lat_decl) * (1 + k * altitude)
                total_radiation += max(0, daily_radiation)
            return total_radiation / 365  # Promedio anual

        # Cargar el archivo GeoJSON
        #geojson_file = "mexicoHigh.json"
        geojson_file = "mx.json"
        
        try:
            gdf = gpd.read_file(geojson_file)
        except Exception as e:
            st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
            st.stop()

        # Agregar el campo de altitud al GeoDataFrame
        gdf["Altitud"] = gdf["name"].map(altitudes)

        # Calcular la radiación promedio anual para cada estado
        gdf["Radiación Anual (W/m²)"] = gdf.apply(
            lambda row: calculate_annual_radiation(row.geometry.centroid.y, row["Altitud"]), axis=1
        )

        # Crear el mapa centrado en México
        mapa = folium.Map(location=[23.6345, -102.5528], zoom_start=5)

        # Agregar una capa de color basada en la radiación promedio anual
        folium.Choropleth(
            geo_data=gdf,
            name="Radiación Solar Promedio Anual",
            data=gdf,
            columns=["name", "Radiación Anual (W/m²)"],
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Radiación Promedio Anual (W/m²)"
        ).add_to(mapa)

        # Mostrar el mapa en Streamlit
        st.title("Mapa de Radiación Solar Promedio Anual en México")
        st.write("""
        Este mapa muestra la radiación solar promedio anual estimada para cada estado de México,
        considerando la latitud, altitud y variaciones estacionales en el ángulo solar.
        """)
        st_folium(mapa, width=800, height=600)


        # Verificar valores calculados de radiación para diferentes ciudades
        import numpy as np
        import pandas as pd

        # Parámetros de la fórmula
        S0 = 1361  # Constante solar (W/m²)
        Ta = 0.75  # Transmisión atmosférica promedio
        k = 0.12   # Incremento de radiación por km de altitud

        # Altitudes promedio estimadas por estado (en kilómetros)
        altitudes = {
            "Aguascalientes": 1.88, "Baja California": 0.58, "Baja California Sur": 0.40,
            "Campeche": 0.10, "Chiapas": 0.72, "Chihuahua": 1.49, "Ciudad de México": 2.24,
            "Coahuila": 1.12, "Colima": 0.33, "Durango": 1.88, "Guanajuato": 1.96,
            "Guerrero": 0.60, "Hidalgo": 1.90, "Jalisco": 1.56, "Estado de México": 2.57,
            "Michoacán": 1.75, "Morelos": 1.66, "Nayarit": 0.70, "Nuevo León": 1.57,
            "Oaxaca": 1.55, "Puebla": 2.13, "Querétaro": 1.82, "Quintana Roo": 0.10,
            "San Luis Potosí": 1.86, "Sinaloa": 0.38, "Sonora": 0.61, "Tabasco": 0.10,
            "Tamaulipas": 0.25, "Tlaxcala": 2.24, "Veracruz": 0.90, "Yucatán": 0.12, "Zacatecas": 2.19
        }

        # Latitudes aproximadas de las ciudades correspondientes
        latitudes = {
            "Aguascalientes": 21.88, "Baja California": 30.33, "Baja California Sur": 24.14,
            "Campeche": 19.83, "Chiapas": 16.75, "Chihuahua": 28.63, "Ciudad de México": 19.43,
            "Coahuila": 27.25, "Colima": 19.12, "Durango": 24.02, "Guanajuato": 21.02,
            "Guerrero": 17.57, "Hidalgo": 20.48, "Jalisco": 20.67, "Estado de México": 19.33,
            "Michoacán": 19.70, "Morelos": 18.93, "Nayarit": 21.50, "Nuevo León": 25.67,
            "Oaxaca": 17.07, "Puebla": 19.05, "Querétaro": 20.59, "Quintana Roo": 19.18,
            "San Luis Potosí": 22.15, "Sinaloa": 25.79, "Sonora": 29.15, "Tabasco": 17.99,
            "Tamaulipas": 23.74, "Tlaxcala": 19.32, "Veracruz": 19.18, "Yucatán": 20.97, "Zacatecas": 22.77
        }

        # Función para calcular la radiación solar
        def calculate_radiation(latitude, altitude):
            radiation = S0 * Ta * np.cos(np.radians(latitude)) * (1 + k * altitude)
            return max(0, radiation)  # Asegurarnos de que no haya valores negativos

        # Calcular radiación para todas las ciudades
        results = []
        for city, lat in latitudes.items():
            altitude = altitudes.get(city, 0)
            radiation = calculate_radiation(lat, altitude)
            results.append({"Ciudad": city, "Latitud (°)": lat, "Altitud (km)": altitude, "Radiación (W/m²)": radiation})

        # Crear un DataFrame con los resultados
        df_results = pd.DataFrame(results)
        df_results_sorted = df_results.sort_values(by="Radiación (W/m²)")
        df_results_sorted


    
######################################33
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Parámetros de los archivos ACE2
tile_size = (1800, 1800)  # Dimensiones de cada mosaico
resolution = 1 / 120  # Resolución de 30 arcsecs en grados
tile_extent = 15  # Cada mosaico cubre 15x15 grados

# Definir los archivos y sus posiciones
files = {
    "00N090W_LAND_30S.ACE2": (0, -90),
    "00N105W_LAND_30S.ACE2": (0, -105),
    "00N120W_LAND_30S.ACE2": (0, -120),
    "15N090W_LAND_30S.ACE2": (15, -90),
    "15N105W_LAND_30S.ACE2": (15, -105),
    "15N120W_LAND_30S.ACE2": (15, -120),
    "30N090W_LAND_30S.ACE2": (30, -90),
    "30N105W_LAND_30S.ACE2": (30, -105),
    "30N120W_LAND_30S.ACE2": (30, -120),
}

# Crear una grilla global para México
min_lat, max_lat = 0, 45  # Límites aproximados de latitud para México
min_lon, max_lon = -120, -75  # Límites aproximados de longitud para México
lat_points = int((max_lat - min_lat) / resolution)
lon_points = int((max_lon - min_lon) / resolution)
global_elevation = np.full((lat_points, lon_points), -32768, dtype=np.float32)  # Usar el valor de vacío

def read_ace2(file_path):
    """Leer un archivo ACE2 como una matriz NumPy."""
    return np.fromfile(file_path, dtype=np.float32).reshape(tile_size)

# Combinar los mosaicos en la grilla global
for file, (sw_lat, sw_lon) in files.items():
    try:
        # Leer los datos del mosaico
        data = read_ace2(file)
        
        # Validar dimensiones del archivo
        if data.shape != tile_size:
            st.warning(f"Dimensiones inesperadas en el archivo {file}")
            continue

        # Calcular las posiciones en la grilla global
        lat_start = int((max_lat - sw_lat - tile_extent) / resolution)
        lat_end = lat_start + tile_size[0]
        lon_start = int((sw_lon - min_lon) / resolution)
        lon_end = lon_start + tile_size[1]

        # Verificar si las coordenadas encajan correctamente
        if (
            lat_start < 0 or lat_end > global_elevation.shape[0] or
            lon_start < 0 or lon_end > global_elevation.shape[1]
        ):
            st.warning(f"Archivo {file} fuera de los límites de la grilla")
            continue

        # Asignar los datos del mosaico
        global_elevation[lat_start:lat_end, lon_start:lon_end] = data
    except Exception as e:
        st.error(f"Error al leer el archivo {file}: {e}")

# Visualizar el mapa en Streamlit
st.title("Mapa de Elevación de México")
st.write("Este mapa muestra la elevación combinada de varios mosaicos ACE2.")

fig, ax = plt.subplots(figsize=(10, 8))
elevation_masked = np.ma.masked_where(global_elevation == -32768, global_elevation)  # Mascara valores vacíos
cmap = plt.cm.terrain
cmap.set_bad(color="white")  # Colorear los valores vacíos en blanco
elevation_plot = ax.imshow(
    elevation_masked,
    extent=(min_lon, max_lon, min_lat, max_lat),
    cmap=cmap,
    origin="upper"
)
plt.colorbar(elevation_plot, ax=ax, label="Elevación (m)")
ax.set_title("Elevación Topográfica de México")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")

st.pyplot(fig)


#import numpy as np
#import plotly.graph_objects as go
#import streamlit as st

# Parámetros de los archivos ACE2
#tile_size = (1800, 1800)  # Dimensiones de cada mosaico
#resolution = 1 / 120  # Resolución de 30 arcsecs en grados
#tile_extent = 15  # Cada mosaico cubre 15x15 grados

# Definir los archivos y sus posiciones
#files = {
#    "00N090W_LAND_30S.ACE2": (0, -90),
#    "00N105W_LAND_30S.ACE2": (0, -105),
#    "00N120W_LAND_30S.ACE2": (0, -120),
#    "15N090W_LAND_30S.ACE2": (15, -90),
#    "15N105W_LAND_30S.ACE2": (15, -105),
#    "15N120W_LAND_30S.ACE2": (15, -120),
#    "30N090W_LAND_30S.ACE2": (30, -90),
#    "30N105W_LAND_30S.ACE2": (30, -105),
#    "30N120W_LAND_30S.ACE2": (30, -120),
#}

#def read_ace2(file_path):
#    """Leer un archivo ACE2 como una matriz NumPy."""
#    return np.fromfile(file_path, dtype=np.float32).reshape(tile_size)

#def plot_tile(data, sw_lat, sw_lon):
#    """Graficar un mosaico ACE2 con Plotly."""
#    latitudes = np.linspace(sw_lat + tile_extent, sw_lat, data.shape[0])
#    longitudes = np.linspace(sw_lon, sw_lon + tile_extent, data.shape[1])
#    masked_data = np.ma.masked_where(data <= 0, data)  # Mascara valores no válidos

#    # Calcular el rango de valores positivos
#    valid_min = masked_data.min()
#    valid_max = masked_data.max()

#    fig = go.Figure(data=go.Heatmap(
#        z=masked_data,
#        x=longitudes,
#        y=latitudes,
#        colorscale="viridis",
#        zmin=valid_min,  # Límite inferior de la escala
#        zmax=valid_max,  # Límite superior de la escala
#        colorbar=dict(title="Elevación (m)")
#    ))
#    fig.update_layout(
#        title=f"Mosaico ({sw_lat}°, {sw_lon}°)",
#        xaxis_title="Longitud",
#        yaxis_title="Latitud",
#        height=700
#    )
#    return fig

## Crear una interfaz en Streamlit
#st.title("Visualización de Mosaicos de Elevación ACE2")
#st.write("Selecciona un mosaico para visualizar su elevación a máxima resolución.")

# Menú desplegable para seleccionar un mosaico
#selected_file = st.selectbox("Selecciona un mosaico:", list(files.keys()))

#if selected_file:
#    # Cargar y graficar el mosaico seleccionado
#    sw_lat, sw_lon = files[selected_file]
#    try:
#        tile_data = read_ace2(selected_file)
#        st.write(f"Visualizando el mosaico: **{selected_file}**")
#        fig = plot_tile(tile_data, sw_lat, sw_lon)
#        st.plotly_chart(fig, use_container_width=True)
#    except Exception as e:
#        st.error(f"Error al cargar el mosaico {selected_file}: {e}")

import numpy as np
import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

# Parámetros para archivos ACE2
tile_size = (1800, 1800)
resolution = 1 / 120  # 30 arcsecs en grados
tile_extent = 15  # Grados cubiertos por cada mosaico

# Archivos ACE2 y sus posiciones
files = {
    "00N090W_LAND_30S.ACE2": (0, -90),
    "00N105W_LAND_30S.ACE2": (0, -105),
    "00N120W_LAND_30S.ACE2": (0, -120),
    "15N090W_LAND_30S.ACE2": (15, -90),
    "15N105W_LAND_30S.ACE2": (15, -105),
    "15N120W_LAND_30S.ACE2": (15, -120),
    "30N090W_LAND_30S.ACE2": (30, -90),
    "30N105W_LAND_30S.ACE2": (30, -105),
    "30N120W_LAND_30S.ACE2": (30, -120),
}

# Crear grilla global para México
min_lat, max_lat = 0, 45
min_lon, max_lon = -120, -75
lat_points = int((max_lat - min_lat) / resolution)
lon_points = int((max_lon - min_lon) / resolution)
global_elevation = np.full((lat_points, lon_points), -32768, dtype=np.float32)

def read_ace2(file_path):
    """Leer archivo ACE2 y convertir a matriz NumPy."""
    return np.fromfile(file_path, dtype=np.float32).reshape(tile_size)

# Combinar mosaicos
for file, (sw_lat, sw_lon) in files.items():
    try:
        data = read_ace2(file)
        lat_start = int((max_lat - sw_lat - tile_extent) / resolution)
        lat_end = lat_start + tile_size[0]
        lon_start = int((sw_lon - min_lon) / resolution)
        lon_end = lon_start + tile_size[1]
        global_elevation[lat_start:lat_end, lon_start:lon_end] = data
    except Exception as e:
        st.warning(f"Error con el archivo {file}: {e}")

# Parámetros para radiación solar
S0 = 1361  # Constante solar (W/m²)
Ta = 0.75  # Transmisión atmosférica promedio
k = 0.12   # Incremento de radiación por km de altitud

def calculate_annual_radiation(latitude, altitude):
    """Calcular radiación solar promedio anual."""
    total_radiation = 0
    for day in range(1, 366):
        declination = 23.45 * np.sin(np.radians((360 / 365) * (day - 81)))
        sin_lat_decl = np.sin(np.radians(latitude)) * np.sin(np.radians(declination))
        cos_lat_decl = np.cos(np.radians(latitude)) * np.cos(np.radians(declination))
        daily_radiation = S0 * Ta * (sin_lat_decl + cos_lat_decl) * (1 + k * altitude)
        total_radiation += max(0, daily_radiation)
    return total_radiation / 365

# Cargar archivo GeoJSON
geojson_file = "mx.json"  # Sustituir con tu archivo
try:
    gdf = gpd.read_file(geojson_file)
except Exception as e:
    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
    st.stop()

# Calcular elevación promedio y radiación por estado
def calculate_state_radiation(row):
    state_shape = row.geometry
    minx, miny, maxx, maxy = state_shape.bounds
    lat_indices = slice(
        int((max_lat - maxy) / resolution),
        int((max_lat - miny) / resolution)
    )
    lon_indices = slice(
        int((minx - min_lon) / resolution),
        int((maxx - min_lon) / resolution)
    )
    state_elevation = global_elevation[lat_indices, lon_indices]
    avg_altitude = np.mean(state_elevation[state_elevation > 0]) / 1000  # km
    latitude = state_shape.centroid.y
    return calculate_annual_radiation(latitude, avg_altitude)

gdf["Radiación Promedio"] = gdf.apply(calculate_state_radiation, axis=1)

# Crear mapa
mapa = folium.Map(location=[23.6345, -102.5528], zoom_start=5)
folium.Choropleth(
    geo_data=gdf,
    name="Radiación Solar",
    data=gdf,
    columns=["name", "Radiación Promedio"],
    key_on="feature.properties.name",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Radiación Promedio Anual (W/m²)"
).add_to(mapa)

# Mostrar mapa en Streamlit
st.title("Mapa Detallado de Radiación Solar Promedio en México")
st_folium(mapa, width=800, height=600)

import os  # Importación para manejar archivos y rutas
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import gdown

# Parámetros del archivo ACE2
file_url = "https://drive.google.com/uc?id=1LcpoOmi-jOX_CyVvdqmGh19X5gVwPmjr"  # URL del archivo para Colima
file_path = "Colima_ACE2.ace2"  # Nombre local del archivo
tile_size = (6000, 6000)  # Dimensiones del mosaico ACE2 (9 arcseconds)

# Descargar el archivo si no existe
if not os.path.exists(file_path):
    st.write("Descargando el archivo ACE2 para Colima...")
    gdown.download(file_url, file_path, quiet=False)

# Leer el archivo ACE2
def read_ace2(file_path, tile_size):
    """Leer un archivo ACE2 y convertirlo en una matriz NumPy."""
    return np.fromfile(file_path, dtype=np.float32).reshape(tile_size)

try:
    elevation_data = read_ace2(file_path, tile_size)
    st.write("Datos de elevación cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar el archivo ACE2: {e}")
    st.stop()

# Máscara para valores no válidos
elevation_masked = np.ma.masked_where(elevation_data <= 0, elevation_data)

# Visualizar los datos de elevación
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.cm.terrain
cmap.set_bad(color="white")  # Colorear los valores inválidos en blanco
elevation_plot = ax.imshow(
    elevation_masked,
    cmap=cmap,
    origin="upper",
    extent=[-105, -103, 18, 20]  # Aproximado para Colima (ajustar si es necesario)
)
plt.colorbar(elevation_plot, ax=ax, label="Elevación (m)")
ax.set_title("Mapa de Elevación para Colima")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")

# Mostrar el mapa en Streamlit
st.title("Mapa de Elevación para Colima")
st.pyplot(fig)
########################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium
import folium
import gdown

# Parámetros del archivo ACE2
file_url = "https://drive.google.com/uc?id=1LcpoOmi-jOX_CyVvdqmGh19X5gVwPmjr"  # URL del archivo para Colima
file_path = "Colima_ACE2.ace2"  # Nombre local del archivo
tile_size = (6000, 6000)  # Dimensiones del mosaico ACE2 (9 arcseconds)
geojson_path = "Colima.json"  # Archivo GeoJSON de Colima

# Descargar el archivo ACE2 si no existe
if not os.path.exists(file_path):
    st.write("Descargando el archivo ACE2 para Colima...")
    gdown.download(file_url, file_path, quiet=False)

# Leer el archivo ACE2
def read_ace2(file_path, tile_size):
    """Leer un archivo ACE2 y convertirlo en una matriz NumPy."""
    return np.fromfile(file_path, dtype=np.float32).reshape(tile_size)

try:
    elevation_data = read_ace2(file_path, tile_size)
    st.write("Datos de elevación cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar el archivo ACE2: {e}")
    st.stop()

# Máscara para valores no válidos
elevation_masked = np.ma.masked_where(elevation_data <= 0, elevation_data)

# Parámetros para radiación solar
S0 = 1361  # Constante solar (W/m²)
Ta = 0.75  # Transmisión atmosférica promedio
k = 0.12   # Incremento de radiación por km de altitud

def calculate_annual_radiation(latitude, altitude):
    """Calcular radiación solar promedio anual considerando declinación solar y ángulo horario."""
    total_radiation = 0
    for day in range(1, 366):
        # Calcular declinación solar
        declination = 23.45 * np.sin(np.radians((360 / 365) * (day - 81)))
        declination_rad = np.radians(declination)
        
        # Convertir latitud a radianes
        latitude_rad = np.radians(latitude)
        
        # Calcular ángulo horario del amanecer/atardecer
        h_s = np.arccos(-np.tan(latitude_rad) * np.tan(declination_rad))
        
        # Calcular radiación diaria
        daily_radiation = (
            S0 * Ta * (1 + k * altitude) * 
            (np.cos(latitude_rad) * np.cos(declination_rad) * np.sin(h_s) +
             h_s * np.sin(latitude_rad) * np.sin(declination_rad))
        )
        
        total_radiation += max(0, daily_radiation)  # Evitar valores negativos

    return total_radiation / 365  # Promedio anual


# Cargar el archivo GeoJSON de Colima
try:
    gdf = gpd.read_file(geojson_path)
    st.write("GeoJSON cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el archivo GeoJSON: {e}")
    st.stop()

def calculate_municipality_radiation(row):
    bounds = row.geometry.bounds  # Obtener límites del municipio
    min_lon, min_lat, max_lon, max_lat = bounds
    #min_lat, max_lat, min_lon, max_lon = 15, 30, -105, -90

    
    # Asegurarnos de que los índices estén dentro de los límites del mosaico
    lon_indices = slice(
        max(0, int((min_lon + 105) * tile_size[1] / 15)),
        min(tile_size[1], int((max_lon + 105) * tile_size[1] / 15))
    )
    lat_indices = slice(
        max(0, int((20 - max_lat) * tile_size[0] / 15)),
        min(tile_size[0], int((20 - min_lat) * tile_size[0] / 15))
    )

    # Extraer elevación del mosaico
    municipality_elevation = elevation_data[lat_indices, lon_indices]
    if municipality_elevation[municipality_elevation > 0].size == 0:
        return 0  # Valor por defecto en caso de datos vacíos

    avg_altitude = np.mean(municipality_elevation[municipality_elevation > 0]) / 1000  # En km
    latitude = row.geometry.centroid.y
    return calculate_annual_radiation(latitude, avg_altitude)


gdf["Radiación Promedio"] = gdf.apply(calculate_municipality_radiation, axis=1)

# Crear un mapa interactivo
mapa = folium.Map(location=[19.2453, -103.725], zoom_start=8)
folium.Choropleth(
    geo_data=gdf,
    name="Radiación Solar",
    data=gdf,
    columns=["CVE_MUN", "Radiación Promedio"],
    key_on="feature.properties.CVE_MUN",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Radiación Promedio Anual (W/m²)"
).add_to(mapa)

# Mostrar mapa en Streamlit
st.title("Mapa de Radiación Solar Promedio en Colima")
st_folium(mapa, width=800, height=600)

