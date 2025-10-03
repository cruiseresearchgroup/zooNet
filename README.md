# ZooNet
A research framework for modeling zoonotic pathogen outbreaks using transmission models, genetic divergence imputation, and graph-based outbreak forecasting.


<p align="center">
  <img src="assets/framework.jpeg" alt="zooNet framework" width="760">
</p>



This repository integrates three main components:

Transmission Model: computes temperature-, humidity-, precipitation-, and proximity-modulated transmission rates (β).

Quantile Regression Imputation (QRI): imputes genetic divergence between pathogen cases using spatiotemporal and host metadata.

BLUE (Bi-Layer hterogeneous graph fUsion nEtwork) – a graph neural network for spatiotemporal epidemic forecasting.

![ZooNet demo](assets/weeks_all.gif)

## Structure
```
ZooNet/
├── transmissionmodel/              
│   ├── h5n1_beta_modulation.py    
│   ├── get_env_inputs.py           
│   ├── run_beta_from_csv.py        
│   └── output_json/                
│
├── imputedevolutionarydistance/              
│   ├── quantile_modeling_pipeline.py     

│
├── BLUE model/                     
│   ├── FullHeteroGNN.py
│   ├── HeteroGraphNetwork.py
│   ├── MRF.py
│   ├── simple_graph_dataset.py
│   ├── metrics.py
│   ├── Model_metrics.py
│   ├── spectral_simple_main.py
│   └── requirements.txt
│
└── data/                           
```

## Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/ZooNet.git
cd ZooNet

# Create environment (example: CUDA 11.8)
conda create -n zoonet python=3.10
conda activate zoonet

# Install dependencies
pip install -r requirements.txt
```





## Transmission Model 
Computes modulated transmission rates (β) for reported cases.

### Usage

1. Place a CSV in `data/` named `cases_input.csv` with columns:
   - `latitude`, `longitude`, `date` (`YYYY-MM-DD`)
2. Run the batch processor:
```
python transmissionmodel/run_beta_from_csv.py
```
3. Results will be saved to `transmissionmodel/output_json/cases_with_beta.csv
` with a new `beta` column.

### To Change
Edit `get_env_inputs.py` to replace dummy values with real environmental extraction from:



## Quantile Modeling Pipeline for Genetic Divergence Imputation

Trains quantile regression models to impute genetic divergence (K80 distance) between pathogen cases using metadata. Includes feature ablation experiments.

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
- `output_ablation_results.csv` with predicted quantiles, residuals, divergence strata, and metrics (MAE, RMSE, R², MAPE, interval coverage).
- Console summary with validation metrics per stratum.


## BLUE: Bi-Layer heterogeneous graph fusion network
A GNN framework for spatiotemporal outbreak forecasting.

### Installation

```bash
# 1. Clone the repo

# 2. Create environment (CUDA 11.8 example)
$ conda create -n blue python=3.10
$ conda activate blue

# 3. Install PyTorch + PyG (adjust CUDA version if needed)
$ pip install -r requirements.txt  # numpy, pandas, scikit‑learn, tqdm, tensorboard, pytorch, torch_geometric, scatter, etc.
```


### Dataset preparation

BLUE expects **weekly graphs** that already combine all raw data into PyG `HeteroData` pickle files
Each file **must** contain:

* Node types: `"county"`, `"case"`
* Edge types: `(county, spatial, county)`, `(case, genetic, case)`, `(case, assignment, county)`
* Node feature tensors named `x`
* County‑level attributes [`infected count`, 'abundance']
* Case-level attributes ['importance']

### Quick start

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

## Evaluation metrics

The following metrics are computed (see `metrics.py`):

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **Pearson** - Pearson correlations
* **Spearman** - Spearman correlations
* **F1 Score** - F1 Score


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
