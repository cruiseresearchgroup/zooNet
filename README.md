# ZooNet

## Transmission Model 
This module computes temperature-, humidity-, precipitation-, and proximity-modulated transmission rates (β) for location of reported cases.

### Structure
```
transmissionmodel/

├── h5n1_beta_modulation.py       # β calculation logic
├── get_env_inputs.py             # Environmental data extractor
├── run_beta_from_csv.py          # Batch calculator from CSV
└── output_json/                  # Output folder for β results
```

### Usage

1. Place a CSV in `data/` named `cases_input.csv` with columns:
   - `latitude`, `longitude`, `date` (`YYYY-MM-DD`)
2. Run the batch processor:
```bash
python run_beta_from_csv.py
```
3. Results will be saved to `output_json/cases_with_beta.csv` with a new `beta` column.

### To Change
Edit `get_env_inputs.py` to replace dummy values with real environmental extraction from:



## Quantile Modeling Pipeline for Genetic Divergence Imputation

This repository contains code and a sample dataset for training and evaluating a quantile regression model to impute genetic divergence between pathogen cases. The model incorporates temporal, geographic, and host metadata and supports feature ablation experiments.

### Files

- `quantile_modeling_pipeline.py`: Main script that trains a quantile regression model (QRI) and runs ablation analysis across divergence strata.

### Input Format

The input dataset should be a CSV file with the following columns:

| Column              | Type    | Description                                             |
|---------------------|---------|---------------------------------------------------------|
| `k80`               | float   | Pairwise genetic divergence                             |
| `delta_days`        | float   | Time difference between samples (in days)               |
| `family1`, `family2`| string  | Host family names for the two sequences                 |
| `state1`, `state2`  | string  | Geographic regions (e.g., US state codes)               |
| `state_distance_km` | float   | Geographic distance between locations in kilometers     |
| `year`              | int     | Year of sample collection                               |

An example is provided in `input_data_example.csv`.

### Usage

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Then run the pipeline:

```bash
python quantile_modeling_pipeline.py
```

This will train quantile regression models across LOW, MID, and HIGH K80 divergence strata, perform feature ablation analysis, and generate performance metrics.

### Output

The script generates:

- `output_ablation_results.csv`: Contains:
  - True and predicted K80 values at 5%, 50%, and 95% quantiles
  - Residual errors
  - Divergence stratum class
  - Metrics including MAE, RMSE, R², MAPE, and 90% interval coverage

Console output will also include summary statistics and validation metrics for each experiment.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
