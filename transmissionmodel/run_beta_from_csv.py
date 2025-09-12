# run_beta_from_csv.py

import pandas as pd
from h5n1_beta_modulation import calculate_beta_with_regime

#Data file import
df = pd.read_csv("data.csv")  

# Ensure dates are parsed
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Calculate β for each row
beta_list = []
for _, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    date_str = row['date'].strftime('%Y-%m-%d')
    beta = calculate_beta_with_regime(lat, lon, date_str)
    beta_list.append(beta)

# Add β to the dataframe
df['beta'] = beta_list


# Output
df.to_csv("cases_with_beta.csv", index=False)

 