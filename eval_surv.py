import xgboost as xgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from run_test import fix_dtypes, get_leaky_columns
from src.utils.feature_selection import filter_features_by_config
from src.survival.evaluation import evaluate_survival_model
from src.survival.predict import predict_event_probabilities

MODEL_PATH = "checkpoints/survival_aft_incr_20260211_201704/model_y_moved.json"
SURVIVAL_PARQUET = "data/processed_features_survival"
EVENTS = ["y_moved"]
HORIZONS = [1, 3, 5]
MODEL_TYPE = "aft"
DISTRIBUTION = "normal"
SIGMA = 1.0
FEATURE_CONFIG = None  # e.g. "configs/data/survival_features.yaml"
TEST_YEARS = [2023, 2024, 2025]
MAX_EVAL_SAMPLES = 10_000_000  # set to int to cap samples, None for full set
RANDOM_STATE = 42
USE_STRATIFIED_SAMPLING = True
SAMPLES_PER_HORIZON = 100000
POS_FRACTION = 0.5


def build_feature_cols(df, events, feature_config_path=None):
    drop_cols = ['sid', 'year', 'refnis', 'y_moved']
    survival_label_cols = []
    for event in events:
        survival_label_cols.extend([
            f"{event}_duration", f"{event}_event_observed"
        ])
    survival_label_cols.extend(
        [c for c in df.columns if c.endswith("_duration") or c.endswith("_event_observed")]
    )
    leak_cols = get_leaky_columns(df.columns)

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
        and c not in leak_cols
        and c not in survival_label_cols
        and c not in events
    ]

    if feature_config_path is not None:
        feature_cols = filter_features_by_config(
            feature_cols,
            feature_config_path=feature_config_path,
            verbose=True,
        )

    return feature_cols


model = xgb.Booster()
model.load_model(MODEL_PATH)

print('model loaded')

def _update_stratum(
    store,
    seen,
    idxs,
    risk_vals,
    dur_vals,
    evt_vals,
    proba_vals,
    max_size,
    rng,
):
    for i in idxs:
        seen += 1
        if len(store["risk"]) < max_size:
            store["risk"].append(float(risk_vals[i]))
            store["dur"].append(float(dur_vals[i]))
            store["evt"].append(int(evt_vals[i]))
            for h in HORIZONS:
                store["proba"][h].append(float(proba_vals[h][i]))
            continue

        j = rng.randint(0, seen)
        if j < max_size:
            store["risk"][j] = float(risk_vals[i])
            store["dur"][j] = float(dur_vals[i])
            store["evt"][j] = int(evt_vals[i])
            for h in HORIZONS:
                store["proba"][h][j] = float(proba_vals[h][i])
    return seen


def _reservoir_update_rows(
    risk_store,
    dur_store,
    evt_store,
    proba_store,
    risk_vals,
    dur_vals,
    evt_vals,
    proba_vals,
    seen,
    max_size,
    rng,
):
    n_new = len(risk_vals)
    if max_size is None:
        risk_store.extend(risk_vals.tolist())
        dur_store.extend(dur_vals.tolist())
        evt_store.extend(evt_vals.tolist())
        for h in HORIZONS:
            proba_store[h].extend(proba_vals[h].tolist())
        return seen + n_new
    for i in range(n_new):
        seen += 1
        if len(risk_store) < max_size:
            risk_store.append(risk_vals[i])
            dur_store.append(dur_vals[i])
            evt_store.append(evt_vals[i])
            for h in HORIZONS:
                proba_store[h].append(proba_vals[h][i])
            continue

        j = rng.randint(0, seen)
        if j < max_size:
            risk_store[j] = risk_vals[i]
            dur_store[j] = dur_vals[i]
            evt_store[j] = evt_vals[i]
            for h in HORIZONS:
                proba_store[h][j] = proba_vals[h][i]
    return seen


feature_cols = None
dataset = ds.dataset(SURVIVAL_PARQUET, format="parquet")
year_filter = ds.field("year").isin(TEST_YEARS)

sample = None
for batch in dataset.to_batches(filter=year_filter):
    sample = batch.to_pandas().head(1000)
    break
if sample is None or len(sample) == 0:
    for batch in dataset.to_batches():
        sample = batch.to_pandas().head(1000)
        break
feature_cols = build_feature_cols(sample, EVENTS, feature_config_path=FEATURE_CONFIG)
model_feature_names = getattr(model, "feature_names", None)
if model_feature_names:
    feature_cols = [c for c in model_feature_names if c in feature_cols]

duration_col = f"{EVENTS[0]}_duration"
event_col = f"{EVENTS[0]}_event_observed"
needed_cols = feature_cols + [duration_col, event_col]

rng = np.random.RandomState(RANDOM_STATE)
seen = 0
if USE_STRATIFIED_SAMPLING:
    pos_target = int(SAMPLES_PER_HORIZON * POS_FRACTION)
    neg_target = int(SAMPLES_PER_HORIZON - pos_target)
    strat_store = {
        h: {
            "pos": {"risk": [], "dur": [], "evt": [], "proba": {hh: [] for hh in HORIZONS}},
            "neg": {"risk": [], "dur": [], "evt": [], "proba": {hh: [] for hh in HORIZONS}},
            "seen_pos": 0,
            "seen_neg": 0,
        }
        for h in HORIZONS
    }
else:
    risk_store = []
    dur_store = []
    evt_store = []
    proba_store = {h: [] for h in HORIZONS}

for batch in dataset.to_batches(filter=year_filter, columns=needed_cols):
    df = batch.to_pandas()
    if len(df) == 0:
        continue

    df = fix_dtypes(df, feature_cols)
    for col in feature_cols:
        if df[col].isnull().any():
            if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'bool':
                df[col] = df[col].fillna(False)

    X = df[feature_cols]
    preds = predict_event_probabilities(
        model,
        X,
        horizons=HORIZONS,
        model_type=MODEL_TYPE,
        distribution=DISTRIBUTION,
        sigma=SIGMA,
    )

    risk_vals = preds["risk_score"]
    dur_vals = df[duration_col].values
    evt_vals = df[event_col].values
    proba_vals = {h: preds.get(f"prob_{h}yr") for h in HORIZONS}

    if USE_STRATIFIED_SAMPLING:
        for h in HORIZONS:
            y_h = (evt_vals == 1) & (dur_vals <= h)
            pos_idx = np.where(y_h)[0]
            neg_idx = np.where(~y_h)[0]
            store = strat_store[h]
            store["seen_pos"] = _update_stratum(
                store["pos"],
                store["seen_pos"],
                pos_idx,
                risk_vals,
                dur_vals,
                evt_vals,
                proba_vals,
                pos_target,
                rng,
            )
            store["seen_neg"] = _update_stratum(
                store["neg"],
                store["seen_neg"],
                neg_idx,
                risk_vals,
                dur_vals,
                evt_vals,
                proba_vals,
                neg_target,
                rng,
            )
    else:
        seen = _reservoir_update_rows(
            risk_store,
            dur_store,
            evt_store,
            proba_store,
            risk_vals,
            dur_vals,
            evt_vals,
            proba_vals,
            seen,
            MAX_EVAL_SAMPLES,
            rng,
        )

    del df, X, preds

if USE_STRATIFIED_SAMPLING:
    risk_store = []
    dur_store = []
    evt_store = []
    proba_store = {h: [] for h in HORIZONS}
    for h in HORIZONS:
        for key in ("pos", "neg"):
            st = strat_store[h][key]
            risk_store.extend(st["risk"])
            dur_store.extend(st["dur"])
            evt_store.extend(st["evt"])
            for hh in HORIZONS:
                proba_store[hh].extend(st["proba"][hh])

if not risk_store:
    raise RuntimeError("No test data loaded for evaluation.")

predicted_risk = np.asarray(risk_store, dtype=np.float64)
duration_test = np.asarray(dur_store, dtype=np.float64)
event_test = np.asarray(evt_store, dtype=np.int32)
event_proba = {h: np.asarray(proba_store[h], dtype=np.float64) for h in HORIZONS}

metrics = evaluate_survival_model(
    predicted_risk=predicted_risk,
    predicted_proba=event_proba,
    duration_test=duration_test,
    event_test=event_test,
    horizons=HORIZONS,
)

print(metrics)
