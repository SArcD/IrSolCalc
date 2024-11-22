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
        x=[0, arrow_x],
        y=[0, arrow_y],
        z=[0, arrow_z],
        mode="lines+text",
        line=dict(color="blue", width=5),
        text=f"Hora: {selected_hour}h<br>Azimut: {azim:.2f}°<br>Elevación: {elev:.2f}°",
        textposition="top center",
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
    st.write(f"La potencia de radiación solar total recibida es de aproximadamente **{radiation_power:.2f} W/m²**.")
    st.write(f"La potencia de radiación solar UV recibida es de aproximadamente **{0.05*radiation_power:.2f} W/m²**.")

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


    

   

