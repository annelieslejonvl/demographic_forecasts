import numpy as np
import pandas as pd

df = pd.read_csv('df_socioec.csv')

cols_mapper = {}
for col in df.columns:
    if "2020" in col:
        new_col = col.replace("|2020", "")
        cols_mapper[col] = new_col
df = df.rename(columns=cols_mapper)

for col in df.columns:
    if "2020" in col:
        def to_float_or_nan(x):
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return np.nan
                if isinstance(x, str):
                    x = x.strip().replace(",", ".")
                    if x == "":
                        return np.nan
                return float(x)
            except Exception:
                return np.nan  # failed cast -> NaN

        df[col] = df[col].apply(to_float_or_nan).astype("float64").fillna(-1.0)

df.to_csv('df_socioec.csv', index=False)