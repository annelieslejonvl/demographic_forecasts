from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.storagelevel import StorageLevel
from .preprocessing import clean_df_for_modeling
from pyspark.ml.feature import ( StringIndexer, OneHotEncoder, VectorAssembler, Imputer, StandardScaler, PCA )
from .estimators import build_estimator
class PipelineBuilder:
    def build_preprocess(self, df: DataFrame, data_spec, model_spec):
        df_clean, bin_like, cont_cols = clean_df_for_modeling(df, data_spec, model_spec)

        label_col = data_spec.label_col
        cat_cols = data_spec.cat_cols
        features_col = model_spec.features_col

        stages = []

        # --- Imputer
        prep = model_spec.preprocess
        impute_cfg = prep.get("impute", {"enabled": True, "strategy": "median"})
        cont_out_cols = cont_cols
        if cont_cols and bool(impute_cfg.get("enabled", True)):
            cont_out_cols = [f"{c}__imp" for c in cont_cols]
            stages.append(Imputer(
                inputCols=cont_cols,
                outputCols=cont_out_cols,
                strategy=impute_cfg.get("strategy", "median")
            ))

        # --- Categorical encoding
        enc = model_spec.encoding or {}
        handle_invalid = enc.get("handle_invalid", "keep")

        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid=handle_invalid) for c in cat_cols]
        encoder = OneHotEncoder(
            inputCols=[f"{c}__idx" for c in cat_cols],
            outputCols=[f"{c}__oh" for c in cat_cols],
            handleInvalid=handle_invalid,
        )
        stages.extend(indexers)
        stages.append(encoder)

        # --- Scaling / Dim reduction (cont block)
        scaling = model_spec.scaling or {}
        dimred = model_spec.dim_reduction or {}

        use_scaling = bool(scaling.get("enabled", False))
        use_dimred = bool(dimred.get("enabled", False))

        cont_final_col = None
        cont_vec_col = None

        if use_scaling or use_dimred:
            cont_vec_col = scaling.get("input_col", "cont_vec")
            stages.append(VectorAssembler(inputCols=cont_out_cols, outputCol=cont_vec_col, handleInvalid="keep"))

            if use_scaling:
                scaled_col = scaling.get("output_col", "cont_vec_scaled")
                stages.append(StandardScaler(
                    inputCol=cont_vec_col,
                    outputCol=scaled_col,
                    withMean=bool(scaling.get("withMean", False)),
                    withStd=bool(scaling.get("withStd", True)),
                ))
                cont_final_col = scaled_col
            else:
                cont_final_col = cont_vec_col

            if use_dimred:
                k_req = int(dimred.get("k", 10))
                p = len(cont_out_cols)
                k_eff = min(k_req, p)
                if k_eff >= 2:
                    pca_out = dimred.get("output_col", "cont_vec_pca")
                    stages.append(PCA(k=k_eff, inputCol=cont_final_col, outputCol=pca_out))
                    cont_final_col = pca_out
                    self._last_pca_k_eff = k_eff
                    self._last_pca_k_req = k_req
                    self._last_pca_p = p
                else:
                    self._last_pca_k_eff = None
                    self._last_pca_k_req = k_req
                    self._last_pca_p = p

        # --- Final feature assembly
        final_inputs = list(bin_like) + [f"{c}__oh" for c in cat_cols]
        if cont_final_col is not None:
            final_inputs.append(cont_final_col)      # vector
        else:
            final_inputs.extend(cont_out_cols)       # scalar cols

        stages.append(VectorAssembler(inputCols=final_inputs, outputCol=features_col, handleInvalid="keep"))

        preprocess_pipe = Pipeline(stages=stages)

        info = {
            "label_col": label_col,
            "features_col": features_col,
            "cat_cols": cat_cols,
            "bin_like": bin_like,
            "cont_cols": cont_cols,
            "cont_out_cols": cont_out_cols,
            "cont_final_col": cont_final_col,
            "uses_scaling": use_scaling,
            "uses_dimred": use_dimred,
            "pca_k_req": getattr(self, "_last_pca_k_req", None),
            "pca_k_eff": getattr(self, "_last_pca_k_eff", None),
            "pca_input_dim": getattr(self, "_last_pca_p", None),
        }

        return df_clean, preprocess_pipe, info

    def build_estimator(self, model_spec, label_col: str, features_col: str):
        return build_estimator(model_spec, label_col=label_col, features_col=features_col)
