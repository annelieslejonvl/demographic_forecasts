from pyspark.sql import functions as F

def clean_df_for_modeling(df, data_spec, model_spec):
    """
    Cleans:
    - label cast
    - numeric casts
    - binary-like: NULL -> 0
    - continuous: NaN/Inf -> NULL
    - categoricals cast to string
    Returns: (df_clean, bin_like, cont_cols)
    """
    label_col = data_spec.label_col
    cat_cols = data_spec.cat_cols
    num_cols = data_spec.num_cols

    bin_like, cont_cols = data_spec.split_numeric()

    # keep only needed cols
    df0 = df.select([label_col] + num_cols + cat_cols)

    prep = model_spec.preprocess
    label_cast = prep.get("label_cast", "double")
    numeric_cast = prep.get("numeric_cast", "double")
    cat_cast = prep.get("categorical_cast", "string")
    bin_null_to = prep.get("bin_null_to", 0.0)
    nan_to_null = bool(prep.get("nan_to_null", True))
    inf_to_null = bool(prep.get("inf_to_null", True))

    df1 = df0.withColumn(label_col, F.col(label_col).cast(label_cast))

    # binary-like
    for c in bin_like:
        df1 = df1.withColumn(c, F.coalesce(F.col(c), F.lit(bin_null_to)).cast(numeric_cast))

    # continuous
    for c in cont_cols:
        col = F.col(c).cast(numeric_cast)
        expr = col
        if nan_to_null:
            expr = F.when(F.isnan(col), None).otherwise(expr)
        if inf_to_null:
            expr = (
                F.when(col == float("inf"), None)
                 .when(col == float("-inf"), None)
                 .otherwise(expr)
            )
        df1 = df1.withColumn(c, expr)

    # categoricals
    for c in cat_cols:
        df1 = df1.withColumn(c, F.col(c).cast(cat_cast))

    return df1, bin_like, cont_cols
