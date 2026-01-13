from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from xgboost import XGBClassifier

def build_estimator(model_spec, label_col: str, features_col: str):
    m = model_spec.model
    kind = (m.get("type") or "").lower()
    params = dict(m.get("params", {}))

    if kind in ("logistic_regression", "lr", "logistic"):
        est = LogisticRegression(**params)
    elif kind in ("random_forest", "rf", "randomforest"):
        est = RandomForestClassifier(**params)
    elif kind in ("gbt", "gbtclassifier", "gradient_boosted_trees"):
        est = GBTClassifier(**params)
    elif kind in ('xgboost'):
        est = XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model.type='{m.get('type')}'")

    if hasattr(est, "setLabelCol"):
        est = est.setLabelCol(label_col)
    if hasattr(est, "setFeaturesCol"):
        est = est.setFeaturesCol(features_col)
    return est
