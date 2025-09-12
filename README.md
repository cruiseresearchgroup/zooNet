# H5N1 Transmission Rate Modulation Module
This module computes temperature-, humidity-, precipitation-, and proximity-modulated transmission rates (β) for H5N1 avian influenza cases in wild birds

## Structure
```
h5n1_beta_modulation/

├── h5n1_beta_modulation.py       # β calculation logic
├── get_env_inputs.py             # Environmental data extractor
├── run_beta_from_csv.py          # Batch calculator from CSV
├── data/
│   └── cases_input.csv           # CSV file of cases
├── output_json/                  # Output folder for β results
└── README.md
```

## Usage

1. Place a CSV in `data/` named `cases_input.csv` with columns:
   - `latitude`, `longitude`, `date` (`YYYY-MM-DD`)
2. Run the batch processor:
```bash
python run_beta_from_csv.py
```
3. Results will be saved to `output_json/cases_with_beta.csv` with a new `beta` column.

## To Change
Edit `get_env_inputs.py` to replace dummy values with real environmental extraction from:


