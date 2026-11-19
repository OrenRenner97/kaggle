import numpy as np
import polars as pl
import pytest

from kagglib.titanic import features as F


def test_shape_order_and_no_nulls(tiny):
    ctx = F.full_context(tiny)
    st = F.fit(tiny, ctx)
    X = F.transform(tiny, st, ctx)
    assert X.columns == ["PassengerId", *F.FEATURES]
    assert X.null_count().sum_horizontal().item() == 0
    assert all(X[c].dtype == pl.Float32 for c in F.FEATURES)


def test_titles_and_age_fill(tiny):
    st = F.fit(tiny)
    X = F.transform(tiny, st, F.full_context(tiny))
    tom = X.filter(pl.col("PassengerId") == 3)
    assert tom["Title_Master"].item() == 1 and tom["IsChild"].item() == 1 and tom["AgeMissing"].item() == 1
    assert X.filter(pl.col("PassengerId") == 8)["Title_Rare"].item() == 1  # Rev → Rare
    # Miss без возраста получает медиану Miss из архива (20), а не общую
    assert X.filter(pl.col("PassengerId") == 7)["Age"].item() == pytest.approx(20.0)


def test_unseen_title_and_missing_values(tiny):
    st = F.fit(tiny)
    new = tiny.head(1).with_columns(pl.lit("Garcia, Dona. Maria").alias("Name"), pl.lit("female").alias("Sex"),
                                    pl.lit(None, pl.Float64).alias("Fare"), pl.lit(None, pl.String).alias("Embarked"),
                                    pl.lit(99, pl.Int64).alias("PassengerId"), pl.lit("NEW").alias("Ticket"))
    X = F.transform(new, st, F.Context())
    assert X["Title_Rare"].item() == 1
    assert X["Emb_S"].item() == 1
    assert np.isfinite(X["LogFarePerPerson"].item())


def test_group_features_exclude_self(tiny):
    """Собственный исход никогда не попадает в групповые признаки (иначе — утечка)."""
    ctx = F.full_context(tiny)
    X = F.transform(tiny, F.fit(tiny, ctx), ctx)
    smith = X.filter(pl.col("PassengerId") == 1)
    assert smith["GroupKnown"].item() == 2 and smith["GroupSurvRate"].item() == 0.0
    alone = X.filter(pl.col("PassengerId") == 6)
    assert alone["GroupKnown"].item() == 0
    assert alone["GroupSurvRate"].item() == pytest.approx(tiny["Survived"].mean())  # приор


def test_fold_context_hides_validation_outcomes(tiny):
    tr, va = tiny.filter(pl.col("PassengerId") != 2), tiny.filter(pl.col("PassengerId") == 2)
    ctx = F.fold_context(tr, tiny)
    X = F.transform(va, F.fit(tr, ctx), ctx)
    # Анна видит исходы Джона и Тома (из обучающей части), но не свой
    assert X["GroupKnown"].item() == 2
    # размер группы считается по всем пассажирам (это не таргет)
    assert X["TicketGroupSize"].item() == 3


def test_flipping_validation_label_does_not_change_its_features(tiny):
    tr, va = tiny.filter(pl.col("PassengerId") != 4), tiny.filter(pl.col("PassengerId") == 4)
    a = F.transform(va, F.fit(tr), F.fold_context(tr, tiny))
    b = F.transform(va.with_columns(pl.lit(0, pl.Int64).alias("Survived")), F.fit(tr), F.fold_context(tr, tiny))
    assert a.equals(b)


def test_state_roundtrip(tiny):
    st = F.fit(tiny)
    assert F.FeatureState.from_json(st.to_json()) == st


def test_missing_column_fails_loudly(tiny):
    with pytest.raises(ValueError, match="нет колонок"):
        F.transform(tiny.drop("Ticket"), F.fit(tiny))
