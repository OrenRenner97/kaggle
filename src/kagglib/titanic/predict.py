"""Инференс Titanic по пакету артефактов. Зависимости: numpy, polars, onnxruntime (для backend=native — ничего)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from kagglib.common import artifacts
from kagglib.titanic import features as F

BACKENDS = ("onnx", "native", "logreg")


class Predictor:
    def __init__(self, bundle: Path, backend: str = "onnx", verify: bool = True):
        if backend not in BACKENDS:
            raise ValueError(f"backend должен быть одним из {BACKENDS}")
        self.bundle = Path(bundle)
        self.manifest = artifacts.verify(self.bundle) if verify else None
        self.state = F.FeatureState.from_json((self.bundle / "feature_state.json").read_text())
        self.archive = pl.read_csv(self.bundle / "reference_outcomes.csv",
                                   schema_overrides={"PassengerId": pl.Int64, "Ticket": pl.String, "Survived": pl.Int64})
        self.backend = backend
        self.threshold = (self.manifest or {}).get("threshold", 0.5)
        if backend == "native":
            from kagglib.common.native import NativeBinaryModel
            from kagglib.titanic.model import NATIVE_SYMBOL
            self._native = NativeBinaryModel(self.bundle / "libtitanic.so", NATIVE_SYMBOL, len(F.FEATURES))
        else:
            import onnxruntime as ort
            name = "catboost.onnx" if backend == "onnx" else "logreg.onnx"
            self._sess = ort.InferenceSession(str(self.bundle / name), providers=["CPUExecutionProvider"])

    def features(self, passengers: pl.DataFrame) -> pl.DataFrame:
        """Признаки для партии пассажиров (например, манифест рейса) с учётом архива."""
        ctx = F.Context(passengers=pl.concat([self.archive.select("PassengerId", "Ticket"),
                                              passengers.select(pl.col("PassengerId").cast(pl.Int64), "Ticket")]),
                        outcomes=self.archive)
        return F.transform(passengers.with_columns(pl.col("PassengerId").cast(pl.Int64)), self.state, ctx)

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "native":
            return self._native.predict_proba(X)
        (probs,) = self._sess.run(["probabilities"], {self._sess.get_inputs()[0].name: X.astype(np.float32)})
        return np.asarray(probs[:, 1], dtype=np.float64)

    def predict(self, passengers: pl.DataFrame) -> pl.DataFrame:
        feats = self.features(passengers)
        p = self.predict_proba_matrix(F.to_numpy(feats))
        return pl.DataFrame({"PassengerId": feats["PassengerId"].cast(pl.Int64), "p_survived": p,
                             "Survived": (p >= self.threshold).astype(np.int64)})
