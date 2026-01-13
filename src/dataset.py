from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from pyspark.sql import DataFrame, functions as F


from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class DatasetSpec:
    key_cols: List[str]
    split: Dict[str, Any]
    tune: Dict[str, Any]
    final: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetSpec":
        return cls(
            key_cols=list(d["key_cols"]),
            split=dict(d.get("split", {})),
            tune=dict(d.get("tune", {})),
            final=dict(d.get("final", {})),
        )

@dataclass
class DatasetBuilder:
    label_col: str
    key_cols: List[str]
    year_col: str

    # ---------- SPLIT ----------
    def time_split(
        self,
        df: DataFrame,
        train_range: Tuple[int, int],
        valid_range: Optional[Tuple[int, int]],
        test_range: Tuple[int, int],
    ):
        train_df = df.filter(F.col(self.year_col).between(*train_range))

        valid_df = (
            df.filter(F.col(self.year_col).between(*valid_range))
            if valid_range is not None
            else None
        )

        test_df = df.filter(F.col(self.year_col).between(*test_range))
        return train_df, valid_df, test_df

    # ---------- SAMPLING ----------
    def sample_train(
        self,
        train_df: DataFrame,
        sampling_cfg: Optional[Dict[str, Any]],
    ):
        if not sampling_cfg or sampling_cfg.get("type", "none") == "none":
            return train_df, {"sampling": "none"}

        seed = int(sampling_cfg.get("seed", 42))
        t = sampling_cfg["type"]

        pos = train_df.filter(F.col(self.label_col) == 1)
        neg = train_df.filter(F.col(self.label_col) == 0)

        if t == "neg_frac":
            frac = float(sampling_cfg["neg_frac"])
            neg_s = neg.sample(False, frac, seed)
            return pos.unionByName(neg_s), {
                "sampling": "neg_frac",
                "neg_frac": frac,
                "seed": seed,
            }

        if t == "neg_per_pos":
            # ⚠️ gebruikt counts → liever NIET in tune
            n_pos = pos.count()
            n_neg = neg.count()
            target_neg = int(n_pos * float(sampling_cfg["neg_per_pos"]))
            frac = min(1.0, target_neg / max(1, n_neg))
            neg_s = neg.sample(False, frac, seed)
            return pos.unionByName(neg_s), {
                "sampling": "neg_per_pos",
                "neg_per_pos": sampling_cfg["neg_per_pos"],
                "neg_frac": frac,
                "seed": seed,
            }

        raise ValueError(f"Unknown sampling type: {t}")

    # ---------- HARD NEGATIVES ----------
    def hard_negatives(
        self,
        train_df: DataFrame,
        scored_negatives: DataFrame,
        hard_cfg: Dict[str, Any],
        n_pos: int,
    ):
        """
        scored_negatives: DataFrame with columns key_cols + p1
        """
        hard_per_pos = int(hard_cfg.get("hard_per_pos", 10))
        q = float(hard_cfg.get("q", 0.001))

        K = hard_per_pos * n_pos

        thr = scored_negatives.approxQuantile("p1", [1 - q], 0.01)[0]
        hard_keys = (
            scored_negatives
            .filter(F.col("p1") >= thr)
            .select(*self.key_cols)
            .limit(K)
        )

        pos = train_df.filter(F.col(self.label_col) == 1)
        hard_neg = (
            train_df
            .filter(F.col(self.label_col) == 0)
            .join(hard_keys, on=self.key_cols, how="inner")
        )

        return pos.unionByName(hard_neg), {
            "hard_per_pos": hard_per_pos,
            "q": q,
            "K": K,
        }
