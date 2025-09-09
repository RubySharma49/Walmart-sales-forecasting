import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ENGINE = "fastparquet"   # or "pyarrow" if you installed it

coeff_s = pd.read_parquet("Results/exp_2025_08_17a/coefficients_scaled.parquet",   engine=ENGINE)
coeff_u = pd.read_parquet("Results/exp_2025_08_17a/coefficients_unscaled.parquet", engine=ENGINE)
models  = pd.read_parquet("Results/exp_2025_08_17a/models.parquet",                engine=ENGINE)
metrics = pd.read_parquet("Results/exp_2025_08_17a/metrics.parquet",               engine=ENGINE)


print(coeff_u )
#metrics_wide = metrics.pivot_table(index="store", columns="metric", values="value")
#print(metrics_wide.loc[store])  print()