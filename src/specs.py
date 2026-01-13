from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class DataSpec:
    label_col: str
    cat_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)
    drop_cols: List[str] = field(default_factory=list)

    # optional explicit splits (als je later wil, maar niet verplicht)
    bin_cols: Optional[List[str]] = None
    cont_cols: Optional[List[str]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DataSpec":
        return DataSpec(
            label_col=d["label_col"],
            cat_cols=list(d.get("cat_cols", [])),
            num_cols=list(d.get("num_cols", [])),
            drop_cols=list(d.get("drop_cols", [])),
            bin_cols=(list(d["bin_cols"]) if "bin_cols" in d else None),
            cont_cols=(list(d["cont_cols"]) if "cont_cols" in d else None),
        )

    def resolve(self, df) -> "DataSpec":
        """
        - Expand tokens like 'socio_*' to all df columns starting with 'socio_'
        - Drop columns that do not exist
        - Apply drop_cols
        """
        df_cols = set(df.columns)

        def expand(lst: List[str]) -> List[str]:
            out = []
            for x in lst:
                if x == "socio_*":
                    out.extend(sorted([c for c in df_cols if c.startswith("socio_")]))
                else:
                    out.append(x)
            return out

        drop = set(self.drop_cols)

        cat = [c for c in expand(self.cat_cols) if c in df_cols and c not in drop]
        num = [c for c in expand(self.num_cols) if c in df_cols and c not in drop]

        bin_cols = None
        cont_cols = None
        if self.bin_cols is not None:
            bin_cols = [c for c in expand(self.bin_cols) if c in df_cols and c not in drop]
        if self.cont_cols is not None:
            cont_cols = [c for c in expand(self.cont_cols) if c in df_cols and c not in drop]

        return DataSpec(
            label_col=self.label_col,
            cat_cols=cat,
            num_cols=num,
            drop_cols=self.drop_cols,
            bin_cols=bin_cols,
            cont_cols=cont_cols,
        )

    def split_numeric(self) -> tuple[list[str], list[str]]:
        """
        Priority:
        1) if bin_cols/cont_cols explicitly provided -> use them
        2) else infer (lags + *_available treated as binary-like)
        """
        if self.bin_cols is not None or self.cont_cols is not None:
            bin_like = sorted(self.bin_cols or [])
            cont = sorted(self.cont_cols or [c for c in self.num_cols if c not in set(bin_like)])
            return bin_like, cont

        num_cols = self.num_cols
        bin_like = set([c for c in num_cols if c.endswith("_available")])
        for c in num_cols:
            if ("lag" in c) and any(c.startswith(p) for p in ["moved_", "birth1_", "birth2_", "divorce_"]):
                bin_like.add(c)
        if "lag1_available" in num_cols:
            bin_like.add("lag1_available")

        bin_like = sorted(bin_like)
        cont = [c for c in num_cols if c not in set(bin_like)]
        return bin_like, cont


@dataclass(frozen=True)
class ModelSpec:
    features_col: str = "features"
    preprocess: Dict[str, Any] = field(default_factory=dict)
    encoding: Dict[str, Any] = field(default_factory=dict)
    scaling: Dict[str, Any] = field(default_factory=dict)
    dim_reduction: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelSpec":
        return ModelSpec(
            features_col=d.get("features_col", "features"),
            preprocess=dict(d.get("preprocess", {})),
            encoding=dict(d.get("encoding", {})),
            scaling=dict(d.get("scaling", {})),
            dim_reduction=dict(d.get("dim_reduction", {})),
            model=dict(d.get("model", {})),
        )
