import pandas as pd
import numpy as np
from datetime import timedelta

def create_data(start_date):
    df_time = pd.DataFrame(pd.date_range(start=start_date, periods=24, freq="H")).rename(columns={0:"tpep_pickup_hour"})
    df_clusters = pd.read_csv("df_grouped_location.csv").drop(columns=["Unnamed: 0", "sum","max","mean","median"])
    df_time['key'] = 0
    df_clusters['key'] = 0
    #outer merge on common key (e.g. a cross join)
    df = df_time.merge(df_clusters, on='key').drop('key', axis=1)
    return df

def get_weather_data(start_date):
    import openmeteo_requests
    import requests_cache
    import pandas as pd
    from retry_requests import retry
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
    	"latitude": 40.7834,
    	"longitude": -73.9663,
    	"start_date": start_date,
    	"end_date": (pd.to_datetime(start_date)+timedelta(days=1)).strftime('%Y-%m-%d'),
    	"hourly": ["temperature_2m", "precipitation", "rain", "snowfall", "snow_depth", "wind_speed_10m"]
    }
    responses = openmeteo.weather_api(url, params=params)
    
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
    
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
    hourly_snow_depth = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
    
    hourly_data = {"date": pd.date_range(
    	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
    	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
    	freq = pd.Timedelta(seconds = hourly.Interval()),
    	inclusive = "left"
    )}
    
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["rain"] = hourly_rain
    hourly_data["snowfall"] = hourly_snowfall
    hourly_data["snow_depth"] = hourly_snow_depth
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_dataframe = pd.DataFrame(data = hourly_data)
    hourly_dataframe["date"] = hourly_dataframe["date"].apply(lambda x: x.tz_convert(None))
    hourly_dataframe = hourly_dataframe.rename(columns={"date":"tpep_pickup_hour"})
    return hourly_dataframe

def is_holiday(df):
    nyc_holidays = pd.read_excel("nyc_holidays.xlsx")
    df["is_holiday"] = df["tpep_pickup_hour"].dt.normalize().isin(nyc_holidays["Observed"])
    return df["is_holiday"].apply(lambda x: 1 if x==True else 0)

def get_last_recorded_demand(df):
    df["Total_Trips_last_Hour"] = np.nan
    real_data = pd.read_parquet("trips_weather_merged.parquet")
    real_data = real_data.sort_values(by=['LocationID', 'tpep_pickup_hour'], ascending=[True, True])
    real_data['Total_Trips_last_Hour'] = real_data.groupby('LocationID')['Total_Trips'].shift(1)
    # real_data = real_data.sort_values(by=['LocationID', 'tpep_pickup_hour'], ascending=[True, True])
    # real_data['Total_Trips_last_Hour'] = real_data.groupby('LocationID')['Total_Trips'].shift(1)
    # Loop through each LocationID
    for loc in df["LocationID"].unique():
        # Filter and sort by pickup time
        loc_rows = df[df["LocationID"] == loc].sort_values("tpep_pickup_hour")
    
        if not loc_rows.empty:
            # Take the first row (by time)
            first_row = loc_rows.iloc[0]
    
            # Try to find matching row in real_data
            mask = (
                (real_data["LocationID"] == first_row["LocationID"]) &
                (real_data["tpep_pickup_hour"] == first_row["tpep_pickup_hour"])
            )
    
            if real_data[mask].shape[0] > 0:
                trip_value = real_data.loc[mask, "Total_Trips_last_Hour"].values[0]
    
                # Now update in df using the original index of the first_row
                df.at[first_row.name, "Total_Trips_last_Hour"] = trip_value

def feature_engineering_demo(df,start_date):
    import pandas as pd
    import numpy as np
    df_weather = get_weather_data(pd.Timestamp(start_date).strftime('%Y-%m-%d'))
    df = df.merge(df_weather, on="tpep_pickup_hour",how="left")
    df["pickup_hour"] = df["tpep_pickup_hour"].dt.hour
    df["pickup_day"] = df["tpep_pickup_hour"].dt.day
    df["pickup_month"] = df["tpep_pickup_hour"].dt.month
    df["weekday"] = df["tpep_pickup_hour"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [5,6] else 0)
    df["rush_hour"] = df.apply(lambda x: 1 if (x["pickup_hour"] in [6,7,8,9,10,16,17,18,19,20]) & (x["is_weekend"] ==0) else 0, axis=1)
    df["quarter"] = df["tpep_pickup_hour"].dt.quarter
    df['dayofyear'] = df['tpep_pickup_hour'].dt.dayofyear
    df["weekofyear"] = df['tpep_pickup_hour'].dt.isocalendar().week
    df["hour_sen"] = np.sin(2 * np.pi * (df["pickup_hour"]+1)/24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["pickup_hour"]+1)/24)
    df["is_holiday"] = is_holiday(df)
    get_last_recorded_demand(df)
    return df

def create_demo_ready_data(start_date):
    df = create_data(start_date)
    df = feature_engineering_demo(df,start_date)
    return df

def feature_engineering_shuffle(df):
    df["pickup_hour"] = df["tpep_pickup_hour"].dt.hour
    df["pickup_day"] = df["tpep_pickup_hour"].dt.day
    df["pickup_month"] = df["tpep_pickup_hour"].dt.month
    df["weekday"] = df["tpep_pickup_hour"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [5,6] else 0)
    df["rush_hour"] = df.apply(lambda x: 1 if (x["pickup_hour"] in [6,7,8,9,10,16,17,18,19,20]) & (x["is_weekend"] ==0) else 0, axis=1)
    df["quarter"] = df["tpep_pickup_hour"].dt.quarter
    df['dayofyear'] = df['tpep_pickup_hour'].dt.dayofyear
    df["weekofyear"] = df['tpep_pickup_hour'].dt.isocalendar().week
    df["hour_sen"] = np.sin(2 * np.pi * (df["pickup_hour"]+1)/24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["pickup_hour"]+1)/24)
    df["is_holiday"] = is_holiday(df)
    df["Location*Hour"] = df.apply(lambda x: str(x["LocationID"])+" "+str(x["pickup_hour"]),axis=1)
    df2 = pd.read_csv("df_grouped_location.csv")
    df = df.merge(df2, on="LocationID", how="left").drop(columns=["Unnamed: 0", "sum","max","mean","median"])
    df = df.sort_values(by=['LocationID', 'tpep_pickup_hour'], ascending=[True, True])
    df['Total_Trips_minus_1_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(1) #Previous Hour Total Trips
    df['Total_Trips_minus_2_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(2) #Previous -2h Total Trips
    df['Total_Trips_minus_3_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(3) #Previous -3h Total Trips
    df['Total_Trips_minus_4_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(4) #Previous -4h Total Trips
    df['Total_Trips_minus_5_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(5) #Previous -5h Total Trips
    df['Total_Trips_minus_24_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(24) #Previous day Total Trips
    df['Total_Trips_minus_25_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(25) #Previous day Total Trips, previous_hour
    df['Total_Trips_minus_26_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(26) #Previous day Total Trips, previous_hour
    df['Total_Trips_minus_27_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(27) #Previous day Total Trips, previous_hour
    df['Total_Trips_minus_28_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(28) #Previous day Total Trips, previous_hour
    df['Total_Trips_minus_29_Hour'] = df.groupby('LocationID')['Total_Trips'].shift(29) #Previous day Total Trips, previous_hour
    df["Total_Trips"] = df["Total_Trips"].apply(np.log1p)
    df = df.dropna()
    return df

def predict_with_lag(group, model, features):
    group = group.sort_values('tpep_pickup_hour').copy()
    group['Total_Trips'] = None  # Model output is already in log scale

    for i in range(len(group)):
        row = group.loc[[group.index[i]], features].copy()
        predicted_total_trips_log = model.predict(row)[0]
        group.at[group.index[i], 'Total_Trips'] = predicted_total_trips_log

        if i + 1 < len(group):
            # Apply exp to the log scale Total_Trips for the previous hour
            group.at[group.index[i + 1], 'Total_Trips_last_Hour'] = np.exp(predicted_total_trips_log)

    return group

def predict_with_lag_extended(group, model, features, lag_columns=['Total_Trips_minus_1_Hour', 'Total_Trips_minus_2_Hour', 'Total_Trips_minus_3_Hour', 'Total_Trips_minus_4_Hour', 'Total_Trips_minus_5_Hour']):
    group = group.sort_values('tpep_pickup_hour').copy()
    group['Total_Trips'] = None  # Model output is already in log scale

    for i in range(len(group)):
        row = group.loc[[group.index[i]], features].copy()
        predicted_total_trips_log = model.predict(row)[0]
        group.at[group.index[i], 'Total_Trips'] = predicted_total_trips_log

        # Update lag features for the next row if it exists
        if i + 1 < len(group):
            predicted_total_trips = np.exp(predicted_total_trips_log)

            # Shift the lag columns down by one
            for j in range(len(lag_columns) - 1, 0, -1):
                group.at[group.index[i + 1], lag_columns[j]] = group.at[group.index[i], lag_columns[j-1]]

            # Assign the current prediction to the 'minus_1_hour' lag column
            group.at[group.index[i + 1], 'Total_Trips_minus_1_Hour'] = predicted_total_trips

    return group


def get_demand_24h(df):
    df["Total_Trips_24_Anterior"] = np.nan
    real_data = pd.read_parquet("trips_weather_merged.parquet")
    real_data['tpep_pickup_hour'] = pd.to_datetime(real_data['tpep_pickup_hour'])
    real_data = real_data.sort_values(by=['LocationID', 'tpep_pickup_hour'])

    # Creamos un índice para facilitar la búsqueda eficiente
    real_data = real_data.set_index(['LocationID', 'tpep_pickup_hour'])

    for index, row in df.iterrows():
        loc_id = row["LocationID"]
        pickup_time = row["tpep_pickup_hour"]

        if not isinstance(pickup_time, pd.Timestamp):
            try:
                pickup_time = pd.to_datetime(pickup_time)
            except ValueError:
                print(f"No se pudo convertir a datetime: {pickup_time}")
                continue

        pickup_time_24_anterior = pickup_time - pd.Timedelta(hours=24)

        try:
            # Buscar el valor exacto de 24 horas antes
            valor_24_anterior = real_data.loc[(loc_id, pickup_time_24_anterior), "Total_Trips"]
            df.at[index, "Total_Trips_24_Anterior"] = valor_24_anterior
        except KeyError:
            # Si no existe el valor exacto, buscar el valor más cercano ANTERIOR a 24 horas
            lower_bound = pickup_time_24_anterior - pd.Timedelta(minutes=5) # Ajusta la ventana si es necesario
            upper_bound = pickup_time_24_anterior + pd.Timedelta(minutes=5) # Ajusta la ventana si es necesario

            relevant_data = real_data.loc[(loc_id, lower_bound):(loc_id, upper_bound)]
            if not relevant_data.empty:
                # Tomar el valor más reciente dentro de la ventana
                df.at[index, "Total_Trips_24_Anterior"] = relevant_data.iloc[-1]["Total_Trips"]
            else:
                df.at[index, "Total_Trips_24_Anterior"] = np.nan # O algún otro valor por defecto

    return df

def feature_engineering_demo_test(df,start_date):
    import pandas as pd
    import numpy as np
    df_weather = get_weather_data(pd.Timestamp(start_date).strftime('%Y-%m-%d'))
    df = df.merge(df_weather, on="tpep_pickup_hour",how="left")
    df["pickup_hour"] = df["tpep_pickup_hour"].dt.hour
    df["pickup_day"] = df["tpep_pickup_hour"].dt.day
    df["pickup_month"] = df["tpep_pickup_hour"].dt.month
    df["weekday"] = df["tpep_pickup_hour"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [5,6] else 0)
    df["rush_hour"] = df.apply(lambda x: 1 if (x["pickup_hour"] in [6,7,8,9,10,16,17,18,19,20]) & (x["is_weekend"] ==0) else 0, axis=1)
    df["quarter"] = df["tpep_pickup_hour"].dt.quarter
    df['dayofyear'] = df['tpep_pickup_hour'].dt.dayofyear
    df["weekofyear"] = df['tpep_pickup_hour'].dt.isocalendar().week
    df["hour_sen"] = np.sin(2 * np.pi * (df["pickup_hour"]+1)/24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["pickup_hour"]+1)/24)
    df["is_holiday"] = is_holiday(df)
    get_previous_demand_first_row(df)
    get_demand_previous_24_to_29h_fixed(df)
    return df

def create_demo_ready_data_test(start_date):
    df = create_data(start_date)
    df = feature_engineering_demo_test(df,start_date)
    return df

def get_demand_previous_24_to_29h_fixed(df):
    real_data = pd.read_parquet("trips_weather_merged.parquet")
    real_data['tpep_pickup_hour'] = pd.to_datetime(real_data['tpep_pickup_hour'])
    real_data = real_data.sort_values(by=['LocationID', 'tpep_pickup_hour'])
    real_data = real_data.set_index(['LocationID', 'tpep_pickup_hour'])

    # Initialize the new lag columns in df with NaN
    for i in range(24, 30):
        lag_column_name = f"Total_Trips_minus_{i}_Hour"
        df[lag_column_name] = np.nan

    for index, row in df.iterrows():
        loc_id = row["LocationID"]
        pickup_time = row["tpep_pickup_hour"]

        if not isinstance(pickup_time, pd.Timestamp):
            try:
                pickup_time = pd.to_datetime(pickup_time)
            except ValueError:
                print(f"No se pudo convertir a datetime: {pickup_time}")
                continue

        for i in range(24, 30):  # Loop for hours 24 to 29
            lag_hours = i
            pickup_time_lagged = pickup_time - pd.Timedelta(hours=lag_hours)
            lag_column_name = f"Total_Trips_minus_{lag_hours}_Hour"

            try:
                # Buscar el valor exacto para la hora específica
                valor_lagged = real_data.loc[(loc_id, pickup_time_lagged), "Total_Trips"]
                df.at[index, lag_column_name] = valor_lagged
            except KeyError:
                # Si no existe el valor exacto, buscar el valor más cercano ANTERIOR a la hora específica
                lower_bound = pickup_time_lagged - pd.Timedelta(minutes=5) # Ajusta la ventana si es necesario
                upper_bound = pickup_time_lagged + pd.Timedelta(minutes=5) # Ajusta la ventana si es necesario

                relevant_data = real_data.loc[(loc_id, lower_bound):(loc_id, upper_bound)]
                if not relevant_data.empty:
                    # Tomar el valor más reciente dentro de la ventana
                    df.at[index, lag_column_name] = relevant_data.iloc[-1]["Total_Trips"]
                else:
                    df.at[index, lag_column_name] = np.nan # O algún otro valor por defecto

    real_data = real_data.reset_index()
    return df

def get_previous_demand_first_row(df):
    real_data = pd.read_parquet("trips_weather_merged.parquet")
    real_data = real_data.sort_values(by=['LocationID', 'tpep_pickup_hour'], ascending=[True, True])

    # Convert 'tpep_pickup_hour' to datetime in both DataFrames if it's not already
    if 'tpep_pickup_hour' in real_data.columns:
        real_data['tpep_pickup_hour'] = pd.to_datetime(real_data['tpep_pickup_hour'])
    if 'tpep_pickup_hour' in df.columns:
        df['tpep_pickup_hour'] = pd.to_datetime(df['tpep_pickup_hour'])

    # Create a dictionary to store the first row index for each LocationID
    first_row_indices = {}
    for index, row in df.iterrows():
        loc_id = row['LocationID']
        if loc_id not in first_row_indices:
            first_row_indices[loc_id] = index

    for loc_id, first_index in first_row_indices.items():
        first_row_pickup_time = df.loc[first_index, 'tpep_pickup_hour']

        # Filter real_data for the current LocationID and pickup times within the last 5 hours
        time_threshold = first_row_pickup_time - pd.Timedelta(hours=5)
        relevant_real_data = real_data[
            (real_data['LocationID'] == loc_id) &
            (real_data['tpep_pickup_hour'] < first_row_pickup_time) &
            (real_data['tpep_pickup_hour'] >= time_threshold)
        ].sort_values(by='tpep_pickup_hour', ascending=False)

        # Extract the Total_Trips for the previous 1 to 5 hours
        previous_trips = relevant_real_data['Total_Trips'].head(5).tolist()

        # Add the previous hour data to the first row of the current LocationID in df
        for i, trip_value in enumerate(previous_trips):
            df.at[first_index, f'Total_Trips_minus_{i+1}_Hour'] = trip_value

    return df