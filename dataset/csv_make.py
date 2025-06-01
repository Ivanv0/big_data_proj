import pandas as pd

df = pd.read_parquet("daily_weather.parquet")

tmb_df = df[df["city_name"] == "Tambov"][
    ["date", "avg_temp_c", "min_temp_c", "max_temp_c", "precipitation_mm"]
]

tmb_df = tmb_df.dropna()

print(tmb_df)

tmb_df.to_csv("tambov_weather.csv", index=False)