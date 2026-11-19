"""Пакет артефактов: все бэкенды дают одно и то же, эталоны воспроизводятся, подмена файла ловится."""

import numpy as np
import polars as pl
import pytest

from kagglib.titanic import features as F
from kagglib.titanic import model
from kagglib.titanic.predict import Predictor


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, kaggle_data):
    tr, te = kaggle_data
    out = tmp_path_factory.mktemp("titanic")
    model.export(model.train(tr, extra_passengers=te), out, golden_df=te.head(50))
    return out


def test_backends_agree(bundle, kaggle_data):
    _, te = kaggle_data
    onnx, nat = Predictor(bundle, "onnx"), Predictor(bundle, "native")
    X = F.to_numpy(onnx.features(te))
    assert np.abs(onnx.predict_proba_matrix(X) - nat.predict_proba_matrix(X)).max() < 1e-5


def test_golden_reproduces(bundle, kaggle_data):
    _, te = kaggle_data
    p = Predictor(bundle, "onnx")
    g = pl.read_csv(bundle / "golden.csv")
    X = F.to_numpy(p.features(te.head(50)))
    np.testing.assert_allclose(X, F.to_numpy(g), rtol=1e-6)
    np.testing.assert_allclose(p.predict_proba_matrix(X), g["p_catboost"].to_numpy(), atol=1e-5)
    np.testing.assert_allclose(Predictor(bundle, "logreg").predict_proba_matrix(X), g["p_logreg"].to_numpy(), atol=1e-5)


def test_logreg_json_formula_matches_onnx(bundle, kaggle_data):
    import json

    _, te = kaggle_data
    p = Predictor(bundle, "logreg")
    X = F.to_numpy(p.features(te))
    w = json.loads((bundle / "logreg.json").read_text())
    z = ((X - np.array(w["mean"])) / np.array(w["scale"])) @ np.array(w["coef"]) + w["intercept"]
    np.testing.assert_allclose(1 / (1 + np.exp(-z)), p.predict_proba_matrix(X), atol=1e-5)


def test_predict_output(bundle, kaggle_data):
    _, te = kaggle_data
    res = Predictor(bundle).predict(te)
    assert res.height == te.height and res["Survived"].is_in([0, 1]).all()
    assert 0.25 < res["Survived"].mean() < 0.55


def test_tampered_bundle_rejected(bundle, tmp_path):
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(bundle, bad)
    (bad / "feature_state.json").write_text((bad / "feature_state.json").read_text().replace("30.0", "31.0"))
    with pytest.raises(ValueError, match="повреждён"):
        Predictor(bad)
