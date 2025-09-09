# Walmart Weekly Sales Forecasting
End-to-end time series forecasting with SARIMAX and exogenous features.

## Highlights / Features
- Expanding-window cross-validation
- Feature selection (settled on 2 final exogenous features)
- Forecast horizon: 15 weeks
- Metrics: RMSE, sMAPE, R²

## Project Structure
- `src/` – reusable code (data, features, models, viz, utils)
- `scripts/` – entrypoint scripts (cv, forecast, store categorization)
- `configs/` – YAML config files
- `notebooks/` – EDA, feature ablation, storytelling
- `data/` – sample data only
- `models/` – trained models
- `reports/` – figures + metrics tables

## Setup
conda create -n walmart-sales-forecasting python=3.10.8 -y
conda activate walmart-sales-forecasting
pip install -r requirements.txt
pip install -e .

## Usage
# Run expanding-window CV
python scripts/cross_validation.py --config configs/Params_set.yml

# Train on full history
python scripts/final_evaluation.py --config configs/Params_set.yml

# Forecast next 15 weeks
python scripts/store_categorization.py --config configs/Params_set.yml

## Results
Example results of store1 model:

- Median RMSE (sample store): 38041
- sMAPE: 12.3%
- Best feature set:  Month_Start_Flag with dollar impact of $91,008.453

## Data info
- Dataset is not included in full. The complete dataset can be found here (https://www.kaggle.com/datasets/mikhail1681/walmart-sales).

- Sample dataset contains sales of 26 stores with features like Holiday_Flag, Temperature, Fuel_Price, CPI and Unemployment. 

## Feature Selection
- Tested multiple exogenous features with expanding-window CV.
- Month_Start_Flag gave the most stable results across stores with high dollar impact in many stores.
- Larger sets added complexity without consistent error reduction and significant impact.  

