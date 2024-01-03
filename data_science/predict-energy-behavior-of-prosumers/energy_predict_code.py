# %%
import warnings

warnings.filterwarnings("ignore")

import os
import gc
import pickle
import holidays
import datetime
import cudf 
import numpy as np
import cupy as np
import pandas as pd
import polars as pl
import plotly.express as px

from sklearn.ensemble import VotingRegressor
import lightgbm as lgb

OFFLINE = True
accelerator = 'cuda' if OFFLINE else 'gpu' 

# %%
class DataStorage:
    
    root = "/kaggle/input/predict-energy-behavior-of-prosumers"
    if OFFLINE :
        root = "/root/miller/data_science/predict-energy-behavior-of-prosumers"

    data_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
    ]
    client_cols = [
        "product_type",
        "county",
        "eic_count",
        "installed_capacity",
        "is_business",
        "date",
    ]
    gas_prices_cols = ["forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"]
    electricity_prices_cols = ["forecast_date", "euros_per_mwh"]
    forecast_weather_cols = [
        "latitude",
        "longitude",
        "hours_ahead",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "forecast_datetime",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]
    historical_weather_cols = [
        "datetime",
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "latitude",
        "longitude",
    ]
    location_cols = ["longitude", "latitude", "county"]
    target_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
    ]
    date_cols = ['datetime','date',"forecast_datetime",]

    def __init__(self):
        self.df_data = cudf.read_csv(
            os.path.join(self.root, "train.csv"),
            usecols=self.data_cols,
            parse_dates=self.date_cols
        )
        self.df_client = cudf.read_csv(
            os.path.join(self.root, "client.csv"),
            usecols=self.client_cols,
            parse_dates=self.date_cols
        )
        self.df_gas_prices = cudf.read_csv(
            os.path.join(self.root, "gas_prices.csv"),
            usecols=self.gas_prices_cols,
            parse_dates=self.date_cols
        )
        self.df_electricity_prices = cudf.read_csv(
            os.path.join(self.root, "electricity_prices.csv"),
            usecols=self.electricity_prices_cols,
            parse_dates=self.date_cols
        )
        self.df_forecast_weather = cudf.read_csv(
            os.path.join(self.root, "forecast_weather.csv"),
            usecols=self.forecast_weather_cols,
            parse_dates=self.date_cols
        )
        self.df_historical_weather = cudf.read_csv(
            os.path.join(self.root, "historical_weather.csv"),
            usecols=self.historical_weather_cols,
            parse_dates=self.date_cols
        )
        self.df_weather_station_to_county_mapping = cudf.read_csv(
            os.path.join(self.root, "weather_station_to_county_mapping.csv"),
            usecols=self.location_cols,
            parse_dates=self.date_cols
        )

        self.df_data = self.df_data.loc[self.df_data['datetime'] >= pd.to_datetime('2022-01-01')]

        self.df_target = self.df_data[self.target_cols]

        self.schema_data = self.df_data.dtypes
        self.schema_client = self.df_client.dtypes 
        self.schema_gas_prices = self.df_gas_prices.dtypes
        self.schema_electricity_prices = self.df_electricity_prices.dtypes
        self.schema_forecast_weather = self.df_forecast_weather.dtypes
        self.schema_historical_weather = self.df_historical_weather.dtypes
        self.schema_target = self.df_target.dtypes

        self.df_weather_station_to_county_mapping['latitude'] = self.df_weather_station_to_county_mapping['latitude'].astype('float32')
        self.df_weather_station_to_county_mapping['longitude'] = self.df_weather_station_to_county_mapping['longitude'].astype('float32')

    def update_with_new_data(
        self,
        df_new_client,
        df_new_gas_prices,
        df_new_electricity_prices,
        df_new_forecast_weather,
        df_new_historical_weather,
        df_new_target,
    ):
        df_new_client = cudf.from_pandas(
            df_new_client[self.client_cols], schema_overrides=self.schema_client
        )
        df_new_gas_prices = cudf.from_pandas(
            df_new_gas_prices[self.gas_prices_cols],
            schema_overrides=self.schema_gas_prices,
        )
        df_new_electricity_prices = cudf.from_pandas(
            df_new_electricity_prices[self.electricity_prices_cols],
            schema_overrides=self.schema_electricity_prices,
        )
        df_new_forecast_weather = cudf.from_pandas(
            df_new_forecast_weather[self.forecast_weather_cols],
            schema_overrides=self.schema_forecast_weather,
        )
        df_new_historical_weather = cudf.from_pandas(
            df_new_historical_weather[self.historical_weather_cols],
            schema_overrides=self.schema_historical_weather,
        )
        df_new_target = cudf.from_pandas(
            df_new_target[self.target_cols], schema_overrides=self.schema_target
        )

        self.df_client = cudf.concat([self.df_client, df_new_client]).unique(
            ["date", "county", "is_business", "product_type"]
        )
        self.df_gas_prices = cudf.concat([self.df_gas_prices, df_new_gas_prices]).unique(
            ["forecast_date"]
        )
        self.df_electricity_prices = cudf.concat(
            [self.df_electricity_prices, df_new_electricity_prices]
        ).unique(["forecast_date"])
        self.df_forecast_weather = cudf.concat(
            [self.df_forecast_weather, df_new_forecast_weather]
        ).unique(["forecast_datetime", "latitude", "longitude", "hours_ahead"])
        self.df_historical_weather = cudf.concat(
            [self.df_historical_weather, df_new_historical_weather]
        ).unique(["datetime", "latitude", "longitude"])
        self.df_target = cudf.concat([self.df_target, df_new_target]).unique(
            ["datetime", "county", "is_business", "product_type", "is_consumption"]
        )

    def preprocess_test(self, df_test):
        df_test = df_test.rename(columns={"prediction_datetime": "datetime"})
        df_test = cudf.from_pandas(
            df_test[self.data_cols[1:]], schema_overrides=self.schema_data
        )
        return df_test


# %%
class FeaturesGenerator:
    def __init__(self, data_storage):
        self.data_storage = data_storage
        self.estonian_holidays = list(holidays.country_holidays('EE', years=range(2021, 2026)).keys())

    def _add_general_features(self, df_features):
        df_features['dayofyear'] = df_features['datetime'].dt.dayofyear
        df_features['hour'] = df_features['datetime'].dt.hour
        df_features['day'] = df_features['datetime'].dt.day
        df_features['weekday'] = df_features['datetime'].dt.weekday
        df_features['month'] = df_features['datetime'].dt.month
        df_features['year'] = df_features['datetime'].dt.year
        df_features['country_holidays'] = cudf.Series(np.where(df_features['datetime'].dt.strftime("%Y-%m-%d").isin(self.estonian_holidays), 1, 0))
        df_features['segment'] = (df_features['county'].astype(str) + '_' +
                                df_features['is_business'].astype(str) + '_' +
                                df_features['product_type'].astype(str) + '_' +
                                df_features['is_consumption'].astype(str))
        df_features['sin_dayofyear'] = np.sin(np.pi * df_features['dayofyear'] / 183)
        df_features['cos_dayofyear'] = np.cos(np.pi * df_features['dayofyear'] / 183)
        df_features['sin_hour'] = np.sin(np.pi * df_features['hour'] / 12)
        df_features['cos_hour'] = np.cos(np.pi * df_features['hour'] / 12)
        return df_features

    def _add_client_features(self, df_features):
        df_client = self.data_storage.df_client.copy(deep=True)

        df_client['date'] = df_client['date'] + pd.to_timedelta(2, unit='D')
        df_features = df_features.merge(
            df_client[['county', 'is_business', 'product_type', 'date']],
            on=['county', 'is_business', 'product_type', 'date'],
            how='left'
        )
        return df_features

    def _add_forecast_weather_features(self, df_features):
        df_forecast_weather = self.data_storage.df_forecast_weather
        df_weather_station_to_county_mapping = (
            self.data_storage.df_weather_station_to_county_mapping
        )

        df_forecast_weather = (
            df_forecast_weather.rename({"forecast_datetime": "datetime"})
            .filter((cudf.col("hours_ahead") >= 22) & cudf.col("hours_ahead") <= 45)
            .drop("hours_ahead")
            .with_columns(
                cudf.col("latitude").cast(cudf.datatypes.Float32),
                cudf.col("longitude").cast(cudf.datatypes.Float32),
            )
            .join(
                df_weather_station_to_county_mapping,
                how="left",
                on=["longitude", "latitude"],
            )
            .drop("longitude", "latitude")
            .with_columns(
                cudf.col("temperature").ewm_mean(span=150).alias("forecast_temperature_span_150"),
                cudf.col("dewpoint").ewm_mean(span=150).alias("forecast_dewpoint_span_150"),
                cudf.col("cloudcover_high").ewm_mean(span=150).alias("forecast_cloudcover_high_span_150"),
                cudf.col("cloudcover_low").ewm_mean(span=150).alias("forecast_cloudcover_low_span_150"),
                cudf.col("cloudcover_total").ewm_mean(span=150).alias("forecast_cloudcover_total_span_150"),
                cudf.col("10_metre_u_wind_component").ewm_mean(span=150).alias("forecast_10_metre_u_wind_component_span_150"),
                cudf.col("10_metre_v_wind_component").ewm_mean(span=150).alias("forecast_10_metre_v_wind_component_span_150"),
                cudf.col("direct_solar_radiation").ewm_mean(span=150).alias("forecast_direct_solar_radiation_span_150"),
                cudf.col("surface_solar_radiation_downwards").ewm_mean(span=150).alias("forecast_surface_solar_radiation_downwards_span_150"),
                cudf.col("snowfall").ewm_mean(span=150).alias("forecast_snowfall_span_150"),
                cudf.col("total_precipitation").ewm_mean(span=150).alias("forecast_total_precipitation_span_150"),
            )
        )

        df_forecast_weather_date = (
            df_forecast_weather.group_by("datetime").mean().drop("county")
        )

        df_forecast_weather_local = (
            df_forecast_weather.filter(cudf.col("county").is_not_null())
            .group_by("county", "datetime")
            .mean()
        )

        for hours_lag in [0, 7 * 24]:
            df_features = df_features.join(
                df_forecast_weather_date.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag)
                ),
                on="datetime",
                how="left",
                suffix=f"_forecast_{hours_lag}h",
            )
            df_features = df_features.join(
                df_forecast_weather_local.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag)
                ),
                on=["county", "datetime"],
                how="left",
                suffix=f"_forecast_local_{hours_lag}h",
            )

        return df_features

    def _add_historical_weather_features(self, df_features):
        df_historical_weather = self.data_storage.df_historical_weather
        df_weather_station_to_county_mapping = (
            self.data_storage.df_weather_station_to_county_mapping
        )

        df_historical_weather = (
            df_historical_weather.with_columns(
                cudf.col("latitude").cast(cudf.datatypes.Float32),
                cudf.col("longitude").cast(cudf.datatypes.Float32),
            )
            .join(
                df_weather_station_to_county_mapping,
                how="left",
                on=["longitude", "latitude"],
            )
            .drop("longitude", "latitude")
            .with_columns(
                cudf.col("temperature").ewm_mean(span=150).alias("historical_temperature_span_150"),
                cudf.col("dewpoint").ewm_mean(span=150).alias("historical_dewpoint_span_150"),
                cudf.col("cloudcover_high").ewm_mean(span=150).alias("historical_cloudcover_high_span_150"),
                cudf.col("cloudcover_low").ewm_mean(span=150).alias("historical_cloudcover_low_span_150"),
                cudf.col("cloudcover_total").ewm_mean(span=150).alias("historical_cloudcover_total_span_150"),
                cudf.col("windspeed_10m").ewm_mean(span=150).alias("historical_windspeed_10m_span_150"),
                cudf.col("winddirection_10m").ewm_mean(span=150).alias("historical_winddirection_10m_span_150"),
                cudf.col("direct_solar_radiation").ewm_mean(span=150).alias("historical_direct_solar_radiation_span_150"),
                cudf.col("shortwave_radiation").ewm_mean(span=150).alias("historical_shortwave_radiation_span_150"),
                cudf.col("snowfall").ewm_mean(span=150).alias("historical_snowfall_span_150"),
                cudf.col("diffuse_radiation").ewm_mean(span=150).alias("historical_diffuse_radiation_span_150"),
                cudf.col("surface_pressure").ewm_mean(span=150).alias("historical_surface_pressure_span_150"),
            )
        )

        df_historical_weather_date = (
            df_historical_weather.group_by("datetime").mean().drop("county")
        )

        df_historical_weather_local = (
            df_historical_weather.filter(cudf.col("county").is_not_null())
            .group_by("county", "datetime")
            .mean()
        )

        for hours_lag in [2 * 24, 7 * 24]:
            df_features = df_features.join(
                df_historical_weather_date.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag)
                ),
                on="datetime",
                how="left",
                suffix=f"_historical_{hours_lag}h",
            )
            df_features = df_features.join(
                df_historical_weather_local.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag)
                ),
                on=["county", "datetime"],
                how="left",
                suffix=f"_historical_local_{hours_lag}h",
            )

        for hours_lag in [1 * 24]:
            df_features = df_features.join(
                df_historical_weather_date.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag),
                    cudf.col("datetime").dt.hour().alias("hour"),
                )
                .filter(cudf.col("hour") <= 10)
                .drop("hour"),
                on="datetime",
                how="left",
                suffix=f"_historical_{hours_lag}h",
            )

        return df_features

    def _add_target_features(self, df_features):
        df_target = self.data_storage.df_target

        df_target_all_type_sum = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"])
            .sum()
            .drop("product_type")
        )

        df_target_all_county_type_sum = (
            df_target.group_by(["datetime", "is_business", "is_consumption"])
            .sum()
            .drop("product_type", "county")
        )

        for hours_lag in [
            2 * 24,
            3 * 24,
            4 * 24,
            5 * 24,
            6 * 24,
            7 * 24,
            8 * 24,
            9 * 24,
            10 * 24,
            11 * 24,
            12 * 24,
            13 * 24,
            14 * 24,
        ]:
            df_features = df_features.join(
                df_target.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag)
                ).rename({"target": f"target_{hours_lag}h"}),
                on=[
                    "county",
                    "is_business",
                    "product_type",
                    "is_consumption",
                    "datetime",
                ],
                how="left",
            )

        for hours_lag in [2 * 24, 3 * 24, 7 * 24, 14 * 24]:
            df_features = df_features.join(
                df_target_all_type_sum.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag)
                ).rename({"target": f"target_all_type_sum_{hours_lag}h"}),
                on=["county", "is_business", "is_consumption", "datetime"],
                how="left",
            )

            df_features = df_features.join(
                df_target_all_county_type_sum.with_columns(
                    cudf.col("datetime") + cudf.duration(hours=hours_lag)
                ).rename({"target": f"target_all_county_type_sum_{hours_lag}h"}),
                on=["is_business", "is_consumption", "datetime"],
                how="left",
                suffix=f"_all_county_type_sum_{hours_lag}h",
            )

        cols_for_stats = [
            f"target_{hours_lag}h" for hours_lag in [2 * 24, 3 * 24, 4 * 24, 5 * 24]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias(f"target_mean"),
            df_features.select(cols_for_stats)
            .transpose()
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std"),
        )

        for target_prefix, lag_nominator, lag_denomonator in [
            ("target", 24 * 7, 24 * 14),
            ("target", 24 * 2, 24 * 9),
            ("target", 24 * 3, 24 * 10),
            ("target", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 7, 24 * 14),
            ("target_all_county_type_sum", 24 * 2, 24 * 3),
            ("target_all_county_type_sum", 24 * 7, 24 * 14),
        ]:
            df_features = df_features.with_columns(
                (
                    cudf.col(f"{target_prefix}_{lag_nominator}h")
                    / (cudf.col(f"{target_prefix}_{lag_denomonator}h") + 1e-3)
                ).alias(f"{target_prefix}_ratio_{lag_nominator}_{lag_denomonator}")
            )

        return df_features

    def _reduce_memory_usage(self, df_features):
        df_features = df_features.with_columns(cudf.col(cudf.Float64).cast(cudf.Float32))
        return df_features

    def _drop_columns(self, df_features):
        df_features = df_features.drop(
            "date", "datetime", "hour", "dayofyear"
        )
        return df_features

    def _to_pandas(self, df_features, y):
        cat_cols = [
            "county",
            "is_business",
            "product_type",
            "is_consumption",
            "segment",
        ]

        if y is not None:
            df_features = pd.concat([df_features.to_pandas(), y.to_pandas()], axis=1)
        else:
            df_features = df_features.to_pandas()

        df_features = df_features.set_index("row_id")
        df_features[cat_cols] = df_features[cat_cols].astype("category")

        return df_features

    def generate_features(self, df_prediction_items):
        # 假设 df_prediction_items 是一个 cudf.DataFrame 对象
        if 'target' in df_prediction_items.columns:
            y = df_prediction_items['target']
            df_prediction_items = df_prediction_items.drop(columns=['target'])
        else:
            y = None

        df_features = df_prediction_items
        df_features['date'] = df_prediction_items['datetime'].dt.strftime("%Y-%m-%d")

        for add_features in [
            self._add_general_features,
            self._add_client_features,
            self._add_forecast_weather_features, 
            self._add_historical_weather_features,
            self._add_target_features,
            self._reduce_memory_usage,
            self._drop_columns,
        ]:
            df_features = add_features(df_features)

        df_features = self._to_pandas(df_features, y)

        return df_features


# %%
class Model:
    def __init__(self):
        self.model_parameters = {
            "n_estimators": 2500,
            "learning_rate": 0.06,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 15,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device": accelerator,
            "verbose":1,
        }

        self.model_consumption = VotingRegressor(
            [
                (
                    f"consumption_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters, random_state=i),
                )
                for i in range(10)
            ],verbose=True
        )
        self.model_production = VotingRegressor(
            [
                (
                    f"production_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters, random_state=i),
                )
                for i in range(10)
            ],verbose=True
        )

    def fit(self, df_train_features):
        mask = df_train_features["is_consumption"] == 1
        self.model_consumption.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"]
        )

        mask = df_train_features["is_consumption"] == 0
        self.model_production.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"]
        )

    def predict(self, df_features):
        predictions = np.zeros(len(df_features))

        mask = df_features["is_consumption"] == 1
        predictions[mask.values] = self.model_consumption.predict(
            df_features[mask]
        ).clip(0)

        mask = df_features["is_consumption"] == 0
        predictions[mask.values] = self.model_production.predict(
            df_features[mask]
        ).clip(0)

        return predictions


# %%
data_storage = DataStorage() 
features_generator = FeaturesGenerator(data_storage=data_storage)

# %%
df_train_features = features_generator.generate_features(data_storage.df_data)
df_train_features = df_train_features[df_train_features['target'].notnull()]

# %%
model = Model()
model.fit(df_train_features)

# %%
import enefit

env = enefit.make_env()
iter_test = env.iter_test()

# %%
for (
    df_test, 
    df_new_target, 
    df_new_client, 
    df_new_historical_weather,
    df_new_forecast_weather, 
    df_new_electricity_prices, 
    df_new_gas_prices, 
    df_sample_prediction
) in iter_test:

    data_storage.update_with_new_data(
        df_new_client=df_new_client,
        df_new_gas_prices=df_new_gas_prices,
        df_new_electricity_prices=df_new_electricity_prices,
        df_new_forecast_weather=df_new_forecast_weather,
        df_new_historical_weather=df_new_historical_weather,
        df_new_target=df_new_target
    )
    df_test = data_storage.preprocess_test(df_test)
    
    df_test_features = features_generator.generate_features(df_test)
    df_sample_prediction["target"] = model.predict(df_test_features)
    
    env.predict(df_sample_prediction)


