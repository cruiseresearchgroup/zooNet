import numpy as np
from datetime import datetime
from get_env_inputs import get_environmental_inputs  # data fetch function
# Base Transmission Rate
BETA_BASE = 0.4

# Water Proximity
def is_near_water(distance_km, threshold=18.0):
   return distance_km <= threshold

# Resevoir Decay
def reservoir_proximity_modifier(distance_km, scale=50):
   return np.exp(-distance_km / scale)

# Temp Modifier
def beta_modifier_temperature(temp_C, near_water=True):
   if near_water:
       peak_temp = 10
       width = 8
   else:
       peak_temp = 15
       width = 6
   hot_cutoff = 30
   base = np.exp(-((temp_C - peak_temp) ** 2) / (2 * width ** 2))
   return base if temp_C <= hot_cutoff else base * 0.5

# === RH modifier ===
def beta_modifier_humidity(rh_percent, near_water=True):
   peak_rh = 60 if near_water else 50
   width = 20 if near_water else 15
   return np.exp(-((rh_percent - peak_rh) ** 2) / (2 * width ** 2))

# === Precipitation modifieQr ===
def beta_modifier_precipitation(precip_mm, near_water=True):
   peak_precip = 4.0 if near_water else 2.0
   width = 3.0 if near_water else 2.0
   return np.exp(-((precip_mm - peak_precip) ** 2) / (2 * width ** 2))


# Main Calculation Function 
def calculate_beta_with_regime(lat, lon, date_str, beta_0=BETA_BASE):
   """
   Main β function for a case defined by (lat, lon, date)
   Pulls environmental and proximity data from helper function
   """
   # Parse date if needed---hi
   if isinstance(date_str, str):
       date = datetime.strptime(date_str, "%Y-%m-%d").date()
   else:
       date = date_str
   temp_C, rh_percent, precip_mm, dist_to_reservoir_km = get_environmental_inputs(lat, lon, date)
   near_water = is_near_water(dist_to_reservoir_km)
   prox_mod = reservoir_proximity_modifier(dist_to_reservoir_km)
   temp_mod = beta_modifier_temperature(temp_C, near_water)
   rh_mod = beta_modifier_humidity(rh_percent, near_water)
   precip_mod = beta_modifier_precipitation(precip_mm, near_water)
   
   beta_t = beta_0 * prox_mod * temp_mod * rh_mod * precip_mod
   return beta_t