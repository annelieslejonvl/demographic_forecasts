# Databricks runner:
# - loops over (data_yaml, model_yaml) combinations
# - builds pipeline via your PipelineBuilder
# - time-based split
# - fits + evaluates
# - logs to MLflow with meaningful run names + config params + YAML artifacts
#
# Assumes you already created the modules:
#   src/config_loader.py (load_yaml)
#   src/specs.py (DataSpec, ModelSpec)
#   src/pipeline_builder.py (PipelineBuilder)
#
# Metrics:
# - PR AUC (area under precision-recall)
# - ROC AUC
# - Brier score
# - Positive rate, n, etc.
#
# Notes:
# - For very large data: consider sampling for quick iterations or using a smaller holdout year first.
# - Uses Spark's BinaryClassificationEvaluator.
#
# Usage example at bottom.
from src.dataset import DatasetBuilder

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional

import mlflow
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.config_loader import load_yaml
from src.specs import DataSpec, ModelSpec
from src.dataset import DatasetSpec
import src.dataset
print(src.dataset.__file__)
print(src.dataset.DatasetSpec.from_dict)
from src.pipeline_builder import PipelineBuilder

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from typing import Tuple, Optional, Dict, Any, List
import copy
import mlflow
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml import PipelineModel
from src.preprocessing import clean_df_for_modeling



# ---------------------------
# Helpers: meaningful run naming
# ---------------------------
from pyspark.sql import functions as F

def add_inverse_prevalence_weights(
    df,
    label_col: str,
    weight_col: str = "weight",
    pos_label: int = 1,
    neg_label: int = 0,
    cap: float | None = None,   # bv 50.0 om extreme weights af te kappen
):
    # 1 actie: twee sommen ophalen
    stats = df.agg(
        F.sum(F.when(F.col(label_col) == pos_label, F.lit(1)).otherwise(F.lit(0))).alias("n_pos"),
        F.sum(F.when(F.col(label_col) == neg_label, F.lit(1)).otherwise(F.lit(0))).alias("n_neg"),
    ).collect()[0]

    n_pos = float(stats["n_pos"])
    n_neg = float(stats["n_neg"])

    if n_pos == 0:
        raise ValueError("No positives in training set; cannot compute weights.")

    pos_w = n_neg / n_pos  # standaard keuze
    if cap is not None:
        pos_w = min(pos_w, float(cap))

    out = df.withColumn(
        weight_col,
        F.when(F.col(label_col) == pos_label, F.lit(pos_w)).otherwise(F.lit(1.0))
    )
    info = {"n_pos": n_pos, "n_neg": n_neg, "pos_weight": pos_w}
    return out, info



def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def config_signature(data_cfg: dict, model_cfg: dict) -> str:
    payload = json.dumps({"data": data_cfg, "model": model_cfg}, sort_keys=True)
    return _short_hash(payload, 10)

def run_name_from_cfg(data_yaml: str, model_yaml: str, data_cfg: dict, model_cfg: dict) -> str:
    # model
    m = (model_cfg.get("model", {}) or {})
    mtype = (m.get("type") or "model").lower()

    # scaling / dimred
    scaling = (model_cfg.get("scaling", {}) or {})
    dimred = (model_cfg.get("dim_reduction", {}) or {})
    sc = "sc1" if bool(scaling.get("enabled", False)) else "sc0"
    dr = "pca{}".format(dimred.get("k", "")) if bool(dimred.get("enabled", False)) else "pca0"

    # data (core vs extended inferred from file name + presence of socio_*)
    data_tag = os.path.splitext(os.path.basename(data_yaml))[0]
    model_tag = os.path.splitext(os.path.basename(model_yaml))[0]

    sig = config_signature(data_cfg, model_cfg)
    return f"{data_tag}__{model_tag}__{mtype}__{sc}__{dr}__{sig}"



# ---------------------------
# Metrics
# ---------------------------
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

def brier_score(pred_df, label_col: str, prob_col: str = "probability") -> float:
    p1 = vector_to_array(F.col(prob_col))[1]
    return (
        pred_df
        .select(F.avg((p1 - F.col(label_col)) * (p1 - F.col(label_col))).alias("brier"))
        .collect()[0]["brier"]
    )



def f1_at_threshold(pred_df, label_col: str, prob_col: str = "probability", threshold: float = 0.5):
    p1 = vector_to_array(F.col(prob_col))[1]
    dfb = pred_df.select(
        F.col(label_col).cast("int").alias("y"),
        (p1 >= F.lit(threshold)).cast("int").alias("yhat"),
    )

    agg = dfb.select(
        F.sum((F.col("yhat") == 1).cast("int")).alias("pred_pos"),
        F.sum((F.col("y") == 1).cast("int")).alias("pos"),
        F.sum(((F.col("yhat") == 1) & (F.col("y") == 1)).cast("int")).alias("tp"),
    ).collect()[0]

    tp = float(agg["tp"])
    pred_pos = float(agg["pred_pos"])
    pos = float(agg["pos"])

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / pos if pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def best_f1_threshold(pred_df, label_col: str, thresholds=None, prob_col: str = "probability"):
    if thresholds is None:
        thresholds = [i/100 for i in range(1, 100)]  # 0.01..0.99

    best = {"threshold": None, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        m = f1_at_threshold(pred_df, label_col=label_col, prob_col=prob_col, threshold=float(t))
        if m["f1"] > best["f1"]:
            best = {"threshold": float(t), **m}
    return best


def eval_binary(pred_df, label_col: str, prob_col: str = "probability"):
    p1 = vector_to_array(F.col(prob_col))[1]

    roc_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )
    f1m = f1_at_threshold(pred_df, label_col=label_col, threshold=0.5)

    roc = float(roc_eval.evaluate(pred_df))
    pr = float(pr_eval.evaluate(pred_df))
    brier = float(
        pred_df.select(F.avg((p1 - F.col(label_col)) * (p1 - F.col(label_col))).alias("brier")).collect()[0]["brier"]
    )

    agg = pred_df.select(
        F.count("*").alias("n"),
        F.avg(F.col(label_col)).alias("pos_rate"),
        F.avg(p1).alias("mean_pred_p1"),
    ).collect()[0]
    metrics = {
        "auc_roc": roc,
        "auc_pr": pr,
        "brier": brier,
        "n": float(agg["n"]),
        "pos_rate": float(agg["pos_rate"]),
        "mean_pred_p1": float(agg["mean_pred_p1"]),
    }
    metrics.update({ "f1@0.5": f1m["f1"], "precision@0.5": f1m["precision"], "recall@0.5": f1m["recall"] })
    return metrics



# ---------------------------
# MLflow logging helpers
# ---------------------------


def log_cfg_as_artifact(path: str, artifact_subdir: str):
    # MLflow wants a local file path; in Databricks, local driver FS is fine.
    mlflow.log_artifact(path, artifact_path=artifact_subdir)

def flatten_dict(d: dict, prefix: str = "") -> Dict[str, str]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif isinstance(v, list):
            # store short lists; long lists -> store counts
            if len(v) <= 30:
                out[key] = json.dumps(v)
            else:
                out[key] = f"list(len={len(v)})"
        else:
            out[key] = str(v)
    return out

def _apply_param_overrides(model_spec, overrides: Dict[str, Any]):
    ms = copy.deepcopy(model_spec)
    ms.model.setdefault("params", {})
    ms.model["params"].update(overrides or {})
    return ms


# ---------------------------
# Runner
# ---------------------------
def run_experiments(
    df,
    experiments_name: str,
    runs,
    year_col: str = "year",
    train_range: Tuple[int, int] = (2012, 2022),
    valid_range: Optional[Tuple[int, int]] = None,
    test_range: Tuple[int, int] = (2023, 2025),
    fit = False,
    # speed knobs
    cache_clean_df: bool = False,
    persist_features: bool = True,

    # reuse preprocess
    preprocess_path: Optional[str] = None,
    save_preprocess_path: Optional[str] = None,

    # hyperparams
    override_params_list: Optional[List[Dict[str, Any]]] = None,

    # NEW: two-phase sampling
    mode: str = "tune",  # "tune" or "final"
    tuning_sampling: Optional[Dict[str, Any]] = None,
    final_sampling: Optional[Dict[str, Any]] = None,
):
    """
    mode:
      - "tune": apply tuning_sampling on TRAIN only (fast), validate on valid/test (unsampled)
      - "final": apply final_sampling on TRAIN only, fit best config on bigger data
    """
    mlflow.set_experiment(experiments_name)
    builder = PipelineBuilder()
   
    param_sets = override_params_list or [{}]
    results = []

    
    for data_yaml, model_yaml,dataset_yaml in runs:
        data_cfg_raw = load_yaml(data_yaml)
        model_cfg_raw = load_yaml(model_yaml)
        dataset_cfg_raw = load_yaml(dataset_yaml)
        data_spec = DataSpec.from_dict(data_cfg_raw).resolve(df)
        model_spec_base = ModelSpec.from_dict(model_cfg_raw)
        ds_spec = DatasetSpec.from_dict(d= dataset_cfg_raw)
        label_col = data_spec.label_col
        ds_builder = DatasetBuilder(label_col = data_spec.label_col , 
                                key_cols = ds_spec.key_cols, 
                                year_col = ds_spec.split.get("year_col", "year")

                                )
                                

        
        # split once
        train_df, valid_df, test_df = ds_builder.time_split(
            df,
            tuple(ds_spec.split["train_range"]),
            tuple(ds_spec.split["valid_range"]) if ds_spec.split.get("valid_range") else None,
            tuple(ds_spec.split["test_range"]),
        )
        mode_cfg = ds_spec.tune if mode == "tune" else ds_spec.final

        sampling_cfg = mode_cfg.get("sampling", {"type": "none"})
        hard_cfg = mode_cfg.get("hard_negatives", {"enabled": False})
        # 2) sampling (tune of final)
        train_df_s, samp_info = ds_builder.sample_train(
            train_df,
            sampling_cfg
        )
        # 3) (optioneel) hard negatives
        if hard_cfg.get("enabled", False):
            # baseline preprocess + model for scoring
            df_clean_tmp, pre_pipe_tmp, info_tmp = builder.build_preprocess(
                train_df_s, data_spec, model_spec_base
            )
            pre_tmp = pre_pipe_tmp.fit(df_clean_tmp)

            base_est = builder.build_estimator(
                model_spec_base,
                label_col=label_col,
                features_col=info_tmp["features_col"],
            )
            base_model = base_est.fit(
                pre_tmp.transform(df_clean_tmp).select(label_col, info_tmp["features_col"])
            )

            scored_neg = score_negatives(   # helper, returns sid, year, p1
                train_df_s, pre_tmp, base_model,
                data_spec, model_spec_base,
                key_cols=ds_spec.key_cols,
            )

            n_pos = train_df_s.filter(F.col(label_col) == 1).count()

            train_df_final, hard_info = ds_builder.hard_negatives(
                train_df_s, scored_neg, hard_cfg, n_pos
            )
        else:
            train_df_final = train_df_s
            # ---- PREPROCESSING (fit op train_df_final) ----
        df_clean_train, preprocess_pipe, info = builder.build_preprocess(
            train_df_final, data_spec, model_spec_base
        )

        # In TUNE mode: meestal NIET laden/saven; in FINAL mode: wel logisch
        if mode == "final" and preprocess_path:
            preprocess_model = PipelineModel.load(preprocess_path)
        else:
            preprocess_model = preprocess_pipe.fit(df_clean_train)
            if mode == "final" and save_preprocess_path:
                preprocess_model.write().overwrite().save(save_preprocess_path)

        features_col = info["features_col"]

        def to_feat(df_part, persist=False):
            df_clean_part, _, _ = clean_df_for_modeling(df_part, data_spec, model_spec_base)
            feat = preprocess_model.transform(df_clean_part).select(label_col, features_col)
            if persist:
                feat = feat.persist(StorageLevel.MEMORY_AND_DISK)
                # géén count() tijdens tuning; te duur
                feat.foreachPartition(lambda _: None)
            return feat
        df_clean_train_w, w_info = add_inverse_prevalence_weights(
            df_clean_train,
            label_col=label_col,
            weight_col="weight",
            cap=50.0,
        )
        # Cache enkel train_feat (wordt hergebruikt over regParams)
        train_feat = preprocess_model.transform(df_clean_train_w).select(label_col, features_col,'weight')
 
        if persist_features:
            train_feat = train_feat.persist(StorageLevel.MEMORY_AND_DISK)
            train_feat.foreachPartition(lambda _: None)

        # Valid/test: in tune vaak NIET persisten (tenzij je ze 20x hergebruikt)
        valid_feat = to_feat(valid_df, persist=False) if valid_df is not None else None
        test_feat  = to_feat(test_df,  persist=False)


        rn_base = run_name_from_cfg(data_yaml, model_yaml, data_cfg_raw, model_cfg_raw)

        for overrides in param_sets:
            model_spec = _apply_param_overrides(model_spec_base, overrides)
            est = builder.build_estimator(model_spec, label_col=label_col, features_col=features_col)
            est_model = est.fit(train_feat)

            pred_train = est_model.transform(train_feat).select(label_col, "probability", "rawPrediction")
            pred_test = est_model.transform(test_feat).select(label_col, "probability", "rawPrediction")

            train_metrics = eval_binary(pred_train, label_col=label_col)
            test_metrics = eval_binary(pred_test, label_col=label_col)

            valid_metrics = None
            if valid_feat is not None:
                pred_valid = est_model.transform(valid_feat).select(label_col, "probability", "rawPrediction")
                valid_metrics = eval_binary(pred_valid, label_col=label_col)

            rn = rn_base if not overrides else f"{rn_base}__" + "__".join([f"{k}={v}" for k, v in overrides.items()])

            with mlflow.start_run(run_name=rn):
                mlflow.log_param("mode", mode)
                mlflow.log_param("data_yaml", data_yaml)
                mlflow.log_param("model_yaml", model_yaml)
                mlflow.log_param("train_range", str(train_range))
                mlflow.log_param("valid_range", str(valid_range) if valid_range else "None")
                mlflow.log_param("test_range", str(test_range))

                # sampling metadata
                for k, v in samp_info.items():
                    mlflow.log_param(k, str(v))

                # overrides
                for k, v in (overrides or {}).items():
                    mlflow.log_param(f"override_{k}", str(v))

                # metrics
                for k, v in train_metrics.items():
                    mlflow.log_metric(f"train_{k}", v)
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                if valid_metrics is not None:
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)

            results.append({
                "run_name": rn,
                "mode": mode,
                **samp_info,
                **{f"override_{k}": v for k, v in (overrides or {}).items()},
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **({} if valid_metrics is None else {f"valid_{k}": v for k, v in valid_metrics.items()}),
            })

        # cleanup
        if persist_features:
            train_feat.unpersist()
            test_feat.unpersist()
            if valid_feat is not None:
                valid_feat.unpersist()
        if cache_clean_df:
            df_clean_train.unpersist()

    return results, train_feat, test_feat, valid_feat, preprocess_model

# ---------------------------
# Example call
# ---------------------------
# runs = [
#   ("config/data_features_core.yaml", "config/model_lr_scaled_pca.yaml"),
#   ("config/data_features_extended.yaml", "config/model_lr_scaled_pca.yaml"),
#   ("config/data_features_core.yaml", "config/model_rf.yaml"),
#   ("config/data_features_extended.yaml", "config/model_rf.yaml"),
# ]
#
# results = run_experiments(
#   demography_synth_features,
#   experiments_name="/Users/<you>/moved_models",
#   runs=runs,
#   year_col="year",
#   train_end_year=2018,
#   valid_year=2019,
#   test_year=2020,
#   cache_clean_df=False,
# )
# results[:2]
