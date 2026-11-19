"""Обучение прод-моделей Titanic и экспорт пакета артефактов.

Основная модель — CatBoost (лучший logloss на CV, нативный экспорт в ONNX и в C++).
Резервная — логистическая регрессия: переносится буквально формулой, служит контрольной точкой.
Выбор и сравнение — в competitions/01_titanic/02_models.ipynb.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from kagglib.common import artifacts, native
from kagglib.titanic import features as F

BUNDLE_VERSION = "titanic-1"
THRESHOLD = 0.5
CATBOOST_PARAMS = {"iterations": 500, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 5, "random_seed": 0,
                   "verbose": 0, "thread_count": 4, "allow_writing_files": False}
LOGREG_C = 0.3
NATIVE_SYMBOL = "titanic_predict_proba"

NATIVE_WRAPPER = r"""
// Обёртка с C ABI над кодом, который сгенерировал CatBoost (catboost_model.cpp).
// Подключается из любого языка с FFI: C, Go (cgo), Rust, Java (JNA/Panama), C#, Python (ctypes).
#include <cmath>
#include <cstddef>
#include <vector>

double ApplyCatboostModel(const std::vector<float>& floatFeatures);

extern "C" {

int titanic_n_features() { return %(n_features)d; }

// X: row-major матрица rows × cols (cols должно быть = titanic_n_features()), out: rows вероятностей.
int titanic_predict_proba(const float* X, std::size_t rows, std::size_t cols, double* out) {
    if (cols != %(n_features)d) return 1;
    std::vector<float> row(cols);
    for (std::size_t i = 0; i < rows; ++i) {
        row.assign(X + i * cols, X + (i + 1) * cols);
        double raw = ApplyCatboostModel(row);
        out[i] = 1.0 / (1.0 + std::exp(-raw));
    }
    return 0;
}

}
"""


def _strip_zipmap(path: Path) -> None:
    """Убирает ZipMap из ONNX: выход `probabilities` становится тензором float[N, 2], а не seq(map).

    seq(map) неудобно читать из C++/C#/JS/Go рантаймов ONNX; тензор читается везде одинаково.
    """
    import onnx

    m = onnx.load(str(path))
    zm = [n for n in m.graph.node if n.op_type == "ZipMap"]
    if not zm:
        return
    src = zm[0].input[0]
    m.graph.node.remove(zm[0])
    for n in m.graph.node:
        n.output[:] = ["probabilities" if o == src else o for o in n.output]
    for o in list(m.graph.output):
        if o.name == "probabilities":
            m.graph.output.remove(o)
    m.graph.output.append(onnx.helper.make_tensor_value_info("probabilities", onnx.TensorProto.FLOAT, ["N", 2]))
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


@dataclass
class Trained:
    state: F.FeatureState
    catboost: object
    logreg: object
    archive: pl.DataFrame  # PassengerId, Ticket, Survived — справочник для групповых признаков
    X: np.ndarray
    y: np.ndarray


def train(train_df: pl.DataFrame, extra_passengers: pl.DataFrame | None = None) -> Trained:
    """Обучает обе модели на всём архиве.

    extra_passengers — пассажиры без исхода, известные на момент обучения (например, манифест рейса):
    нужны только для размеров групп по билету.
    """
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ctx = F.full_context(train_df, extra_passengers)
    state = F.fit(train_df, ctx)
    X = F.to_numpy(F.transform(train_df, state, ctx))
    y = train_df["Survived"].to_numpy()
    cb = CatBoostClassifier(**CATBOOST_PARAMS).fit(X, y)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=LOGREG_C, max_iter=2000)).fit(X, y)
    archive = train_df.select("PassengerId", "Ticket", pl.col("Survived").cast(pl.Int64))
    return Trained(state, cb, lr, archive, X, y)


def export(t: Trained, out: Path, golden_df: pl.DataFrame | None = None, build_native: bool = True) -> Path:
    """Сохраняет пакет артефактов в каталог `out`."""
    from skl2onnx import to_onnx

    out.mkdir(parents=True, exist_ok=True)
    nf = len(F.FEATURES)

    (out / "feature_state.json").write_text(t.state.to_json())
    t.archive.write_csv(out / "reference_outcomes.csv")

    # CatBoost: родной формат, ONNX, C++
    t.catboost.save_model(str(out / "catboost.cbm"))
    t.catboost.save_model(str(out / "catboost.onnx"), format="onnx",
                          export_parameters={"onnx_domain": "ai.catboost", "onnx_model_version": 1,
                                             "onnx_graph_name": BUNDLE_VERSION})
    _strip_zipmap(out / "catboost.onnx")
    t.catboost.save_model(str(out / "catboost_model.cpp"), format="cpp")
    (out / "titanic_native.cpp").write_text(NATIVE_WRAPPER % {"n_features": nf})
    if build_native:
        native.compile_shared([out / "catboost_model.cpp", out / "titanic_native.cpp"], out / "libtitanic.so")

    # LogReg: ONNX + формула в JSON (стандартизация → веса → сигмоида)
    onx = to_onnx(t.logreg, t.X[:1].astype(np.float32), options={id(t.logreg): {"zipmap": False}},
                  target_opset=17)
    (out / "logreg.onnx").write_bytes(onx.SerializeToString())
    scaler, lr = t.logreg[0], t.logreg[-1]
    (out / "logreg.json").write_text(json.dumps({
        "formula": "p = sigmoid(sum(coef[i] * (x[i] - mean[i]) / scale[i]) + intercept)",
        "features": list(F.FEATURES), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
    }, indent=2))

    # Эталонные векторы: чтобы реализация на другой платформе могла проверить себя бит-в-бит
    if golden_df is not None:
        ctx = F.Context(passengers=pl.concat([t.archive.select("PassengerId", "Ticket"),
                                              golden_df.select("PassengerId", "Ticket")]),
                        outcomes=t.archive)
        feats = F.transform(golden_df, t.state, ctx)
        Xg = F.to_numpy(feats)
        feats.with_columns(
            pl.Series("p_catboost", t.catboost.predict_proba(Xg)[:, 1]),
            pl.Series("p_logreg", t.logreg.predict_proba(Xg)[:, 1]),
        ).write_csv(out / "golden.csv")

    artifacts.write_manifest(out, {
        "bundle": BUNDLE_VERSION,
        "task": "titanic: P(Survived=1)",
        "threshold": THRESHOLD,
        "features": list(F.FEATURES),
        "input": f"float32[N, {nf}], порядок = features; признаки считаются kagglib.titanic.features",
        "models": {
            "catboost.onnx": "основная; вход 'features' float32[N,21], выход 'probabilities' float32[N,2], берём [:,1]",
            "libtitanic.so": f"основная, нативно; C ABI: int {NATIVE_SYMBOL}(const float*, size_t rows, size_t cols, double* out)",
            "catboost.cbm": "основная, формат CatBoost",
            "logreg.onnx": "резервная; вход float32[N,21], выход 'probabilities' float32[N,2]",
            "logreg.json": "резервная формулой",
        },
        "golden.csv": "эталон для проверки других реализаций: признаки + p_catboost + p_logreg; партия = первые 50 строк "
                      "test.csv вместе с архивом (TicketGroupSize зависит от состава партии!)",
        "catboost_params": CATBOOST_PARAMS,
    })
    return out
