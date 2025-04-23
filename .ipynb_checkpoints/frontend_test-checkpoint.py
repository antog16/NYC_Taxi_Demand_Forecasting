import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import datetime as dt
from Funciones import *
from datetime import timedelta
import geopandas as gpd
import pydeck as pdk
from matplotlib import cm, colors
import json
import plotly.express as px

# Configuración
st.set_page_config(page_title="Estimador de Demanda de Taxis en NYC", layout="wide")
st.image("images/banner_test_2.jpg", use_container_width=True)
st.title("Estimador de Demanda de Taxis en NYC")
shap.initjs()


tab1, tab2 = st.tabs([
    "🧮 Predicción",
    "📊 Demanda Hora a Hora",
])

with tab1:
    # Fecha inicial y máxima permitida
    fecha_inicial = dt.date(2025, 1, 1)
    fecha_maxima = dt.date(2025, 2, 1)
    
    fecha_seleccionada = st.date_input("Selecciona fecha y hora",value=fecha_inicial, min_value=fecha_inicial, max_value=fecha_maxima)
    
    # Input de hora: solo horas en punto (0, 1, 2, ..., 23)
    horas = [f"{i:02}:00" for i in range(24)]  # Genera las horas 00:00, 01:00, ..., 23:00
    hora_seleccionada_raw = st.selectbox("Selecciona la hora", horas)
    
    # Convertir la hora seleccionada a formato datetime.time
    hora_seleccionada = dt.datetime.strptime(hora_seleccionada_raw, "%H:%M").time()
    fecha_hora_combinada = dt.datetime.combine(fecha_seleccionada, hora_seleccionada)
    
    # Formatea el objeto datetime al formato YYYY-MM-DD HH:MM
    formato_deseado = "%Y-%m-%d %H:%M"
    start_date = fecha_hora_combinada.strftime(formato_deseado)
    
    # Cargar pipeline y dataset
    df = create_demo_ready_data_test(start_date)
    demo_sorted = df.sort_values(['LocationID', 'tpep_pickup_hour']).copy()
    models = {0: joblib.load("modelos/Pipeline_high_demand.pkl"), 1: joblib.load("modelos/Pipeline_low_demand.pkl"), 2: joblib.load("modelos/Pipeline_mid_demand.pkl")}
    
    # Cargar el CSV con los nombres de las zonas
    taxi_zone_lookup = pd.read_csv("data/taxi_zones/taxi_zone_lookup.csv")
    
    # Generación de predicciones
    features = [col for col in demo_sorted.columns if col not in [
        'Total_Trips','tpep_pickup_hour', "cluster"
    ]]
    
    grouped_by_location = demo_sorted.groupby('LocationID')
    predicted_data = []
    
    for loc_id, group in grouped_by_location:
        cluster_id = int(group['cluster'].iloc[0])
        model = models[cluster_id]
        predicted_group = predict_with_lag_extended(group.copy(), model, features)
        predicted_data.append(predicted_group)
    
    demo_sorted_predicted = pd.concat(predicted_data).sort_values(by=['LocationID', 'tpep_pickup_hour']).reset_index(drop=True)
    
    # Select only the desired columns
    demo_sorted_predicted = demo_sorted_predicted[['Total_Trips_minus_1_Hour', 'Total_Trips'] + [col for col in demo_sorted_predicted.columns if col not in ['Total_Trips_minus_1_Hour', 'Total_Trips']]]
    demo_sorted_predicted["Total_Trips"] = demo_sorted_predicted["Total_Trips"].astype("float").apply(np.exp)
    display_dataset_page_1 = demo_sorted_predicted[demo_sorted_predicted["tpep_pickup_hour"]==start_date]
    display_dataset_page_1=display_dataset_page_1.merge(taxi_zone_lookup, on="LocationID", how="left")
    display_dataset_page_1["Total_Trips_minus_1_Hour"] = display_dataset_page_1["Total_Trips_minus_1_Hour"].round(2)
    display_dataset_page_1["Total_Trips"] = display_dataset_page_1["Total_Trips"].round(2)
    

    #Incluimos mapa mostrando la demanda de la hora

    gdf = gpd.read_file("taxi_zones.shp")

    # Set the CRS manually to EPSG:2263 (since it came without one)
    gdf = gdf.set_crs(epsg=2263)

    # Transformar a lat/lon
    gdf = gdf.to_crs(epsg=4326)

    gdf = gdf.join(taxi_zone_lookup).drop(columns="service_zone")

    manhattan_gdf = gdf[gdf["Borough"] == "Manhattan"].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Merge with the predicted data for the selected hour
    manhattan_gdf = manhattan_gdf.merge(display_dataset_page_1[["LocationID", "Total_Trips"]], how="left", on="LocationID")

    # Fill NaN Total_Trips with 0 for zones with no predictions
    manhattan_gdf["Total_Trips"] = manhattan_gdf["Total_Trips"].fillna(0).round(2)

    # Convert GeoDataFrame to GeoJSON
    geojson = json.loads(manhattan_gdf.to_json())

    # Normalize 'Total_Trips' for color and height
    min_val = manhattan_gdf["Total_Trips"].min()
    max_val = manhattan_gdf["Total_Trips"].max()
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    colormap = cm.get_cmap("YlOrRd")


    # Add 'fill_color' and 'elevation' properties to each feature
    for feature in geojson["features"]:
        val = feature["properties"].get("Total_Trips", 0) # Default to 0 if Total_Trips is missing
        rgba = colormap(norm(val), bytes=True)
        feature["properties"]["fill_color"] = [int(rgba[0]), int(rgba[1]), int(rgba[2]), 180]  # semi-transparent
        feature["properties"]["elevation"] = val  # height based on 'Total_Trips'

    # Create a 3D extruded PolygonLayer
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        extruded=True,
        get_elevation="properties.elevation * 10",
        get_fill_color="properties.fill_color",
        get_line_color=[0, 0, 0, 150],
        pickable=True,
        auto_highlight=True,
        get_properties=["Zone", "LocationID", "Total_Trips"]
    )

    # View settings
    view_state = pdk.ViewState(
        latitude=manhattan_gdf.geometry.centroid.y.mean(),
        longitude=manhattan_gdf.geometry.centroid.x.mean(),
        zoom=11,
        pitch=45,  # tilt to view in 3D
        bearing=0,
    )

    # Render deck
    r = pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=view_state,
        tooltip={
        "html": "<b>Zone:</b> {Zone}<br>"
                "<b>LocationID:</b> {LocationID}<br>"
                "<b>Total Trips:</b> {Total_Trips}"
    }
    )

    # Show or export

    #Total de viajes en la siguiente hora
    st.metric(label="Total Viajes Próxima Hora", value=f"{manhattan_gdf["Total_Trips"].sum():.2f}")
    
    st.markdown("### Cantidad De Taxis Por Barrio") 

    #
    st.pydeck_chart(r)


    #Mostramos Dataset
    st.dataframe(display_dataset_page_1[["LocationID","Zone","cluster","Total_Trips"]].sort_values("Total_Trips", ascending=False), use_container_width=True)


with tab2:

    #Cargar datos históricos (train data)
    fecha_hora_combinada_previa = fecha_hora_combinada - dt.timedelta(hours=48)
    df_train = pd.read_parquet("data/train_data/trips_weather_merged.parquet")
    df_train = df_train[(df_train["tpep_pickup_hour"]<=fecha_hora_combinada) & (df_train["tpep_pickup_hour"]>fecha_hora_combinada_previa)]
    
    # Crear un diccionario de mapeo entre LocationID y nombre de zona
    location_dict = taxi_zone_lookup.set_index('LocationID')['Zone'].to_dict()
    
    # Obtengo los valores únicos de la columna 'LocationId' para el multiselect
    location_ids_unicos = sorted(df['LocationID'].unique())
    
    # Agrega un selectbox para que el usuario elija un LocationId mostrando nombre y LocationID
    seleccion_location_id = st.selectbox(
        "Selecciona un LocationId para filtrar:", 
        ['Todos'] + [f"{loc_id} - {location_dict.get(loc_id, 'Unknown')}" for loc_id in location_ids_unicos]
    )

    # Extraer solo el LocationID de la opción seleccionada
    if seleccion_location_id != 'Todos':
        seleccion_location_id = int(seleccion_location_id.split(" - ")[0])
    
    # Filtra el DataFrame basado en la selección del usuario
    if seleccion_location_id == 'Todos':
        df_filtrado_predicted = demo_sorted_predicted
        df_filtrado_train = df_train
    else:
        df_filtrado_predicted = demo_sorted_predicted[demo_sorted_predicted['LocationID'] == seleccion_location_id]
        df_filtrado_train = df_train[df_train['LocationID'] == seleccion_location_id]
                                    
    st.subheader("Tendencia de Total de Viajes a lo largo del Tiempo")

    # Agrupar por 'tpep_pickup_hour' y calcular la suma de 'Total_Trips'
    grouped_predicted = df_filtrado_predicted.groupby('tpep_pickup_hour')['Total_Trips'].sum().reset_index()
    grouped_predicted['Fuente'] = 'Predicciones'
    
    grouped_historical = df_filtrado_train.groupby('tpep_pickup_hour')['Total_Trips'].sum().reset_index()
    grouped_historical['Fuente'] = 'Histórico'
    
    # --- Combinar los DataFrames ---
    combined_data = pd.concat([grouped_historical, grouped_predicted], ignore_index=True)

    # --- Convertir 'tpep_pickup_hour' a datetime para asegurar el orden correcto ---
    try:
        combined_data['tpep_pickup_hour'] = pd.to_datetime(combined_data['tpep_pickup_hour'])
    except Exception as e:
        st.warning(f"No se pudieron convertir las fechas a formato datetime. Asegúrate de que el formato sea consistente. Error: {e}")

    # --- Ordenar los datos por 'tpep_pickup_hour' ---
    combined_data = combined_data.sort_values(by='tpep_pickup_hour')


    # --- Crear el gráfico de líneas con Plotly Express ---
    fig = px.line(combined_data, x="tpep_pickup_hour", y="Total_Trips", color="Fuente",
                  title="Suma Total de Viajes por Hora (Histórico y Predicciones)",
                  labels={'tpep_pickup_hour': 'Hora de Recogida',
                          'Total_Trips': 'Suma Total de Viajes'})

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    demo_sorted_predicted["Total_Trips_minus_1_Hour"] = demo_sorted_predicted["Total_Trips_minus_1_Hour"].round(2)
    demo_sorted_predicted["Total_Trips"] = demo_sorted_predicted["Total_Trips"].round(2)
    demo_sorted_predicted_filtered =demo_sorted_predicted[demo_sorted_predicted["LocationID"]==seleccion_location_id]
    
    if seleccion_location_id == 'Todos':
        demo_sorted_predicted_filtered_grouped = demo_sorted_predicted.groupby("tpep_pickup_hour").agg({
    "Total_Trips_minus_1_Hour": "sum",
    "Total_Trips": "sum"
}).reset_index()
        st.dataframe(demo_sorted_predicted_filtered_grouped, use_container_width=True)
    else:
        st.dataframe(demo_sorted_predicted_filtered[["tpep_pickup_hour","Total_Trips_minus_1_Hour","Total_Trips"]], use_container_width=True)
    