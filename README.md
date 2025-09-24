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

## BLUE: Bi-Layer heterogeneous graph fusion network

## 🗂️ Repository structure

```
BLUE model/
├── HeteroGraphNetwork.py         # BLUE implementation
├── spectral_loss.py              # Spectral Alignment loss implementation
├── laplacian.py                  # Laplacian Matrix implementation
├── MRF.py                        # Markov Random Field smoothing module
├── simple_graph_dataset.py       # Windowed time‑series dataset loader
├── metrics.py                    # Prediction process
├── Model_metrics.py              # MAE / RMSE / PCC /SCC / F1 Score evaluation metrics
├── spectral_simple_main.py       # Train / Val / Eval entry‑point
└── requirements.txt              # environments
```

## ⚙️ Installation

```bash
# 1. Clone the repo

# 2. Create environment (CUDA 11.8 example)
$ conda create -n blue python=3.10
$ conda activate blue

# 3. Install PyTorch + PyG (adjust CUDA version if needed)
$ pip install -r requirements.txt  # numpy, pandas, scikit‑learn, tqdm, tensorboard, pytorch, torch_geometric, scatter, etc.
```

---

## 📄 Dataset preparation

BLUE expects **weekly graphs** that already combine all raw data into PyG `HeteroData` pickle files
Each file **must** contain:

* Node types: `"county"`, `"case"`
* Edge types: `(county, spatial, county)`, `(case, genetic, case)`, `(case, assignment, county)`
* Node feature tensors named `x`
* County‑level attributes [`infected count`, 'abundance']
* Case-level attributes ['importance']

## 🚀 Quick start

```bash
python spectral_simple_main.py \
    --dataset avian \
    --batch_size 4 \
    --window_size 4 \
    --pred_horizon 4 \
    --hidden_dim 16 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.3 \
    --epochs 100 \
    --spectral_gamma 0.9 \
    --loss_type 'infection_weighted' \
    --use_eigenvalue_constraint True \
    --eigenvalue_loss_type cosine_similarity \
    --spectral_k 10 \
    --infection_zero_weight 1.0 \
    --infection_low_weight 8.0 \
    --infection_med_weight 15.0 \
    --infection_high_weight 25.0 \
    --infection_low_threshold 1.0 \
    --infection_med_threshold 5.0 \
    --infection_high_threshold 20.0
```

| Flag             | Meaning                           | Default |
| ---------------- | --------------------------------- | ------- |
| --window_size    | $w$ historic weeks fed to encoder | 4       |
| --pred_horizon   | $h$ weeks to forecast             | 4       |
| --num_mrf        | MRF smoothing layers              | 1       |
| --spectral_gamma | weight of spectral approximation  | 0.9     |

`spectral_simple_main.py --help` prints the full list.

During training the script will output fold‑wise **MAE / RMSE / F1 / Pearson / Spearman** and save the best model to `./save_results/`.

---

## 📊 Evaluation metrics

The following metrics are computed (see `metrics.py`):

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **Pearson** - Pearson correlations
* **Spearman** - Spearman correlations
* **F1 Score** - F1 Score


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
