"""Признаки Titanic. Один и тот же код в ноутбуках, в CV и в проде.

Два вида признаков:
- **по самому пассажиру** (пол, титул, возраст, тариф...) — нужны статистики для заполнения пропусков,
  которые считаются на обучающих данных (`fit`) и сохраняются в артефакт (`FeatureState`);
- **по связям** (размер группы по билету, судьба попутчиков) — зависят от того, кого ещё мы знаем.
  Это задаётся явно через `Context`, чтобы в CV нельзя было случайно подсмотреть исходы валидации.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import polars as pl

TITLES = ("Mr", "Mrs", "Miss", "Master", "Rare")
TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}  # всё остальное (Dr, Rev, Col, Countess, Dona, невиданное) → Rare
PORTS = ("C", "Q", "S")
CHILD_AGE = 14.0
LARGE_FAMILY = 5

FEATURES: tuple[str, ...] = (
    "IsMale", "Pclass",
    *(f"Title_{t}" for t in TITLES),
    "Age", "AgeMissing", "IsChild",
    "FamilySize", "IsAlone", "LargeFamily",
    "TicketGroupSize", "LogFarePerPerson", "HasCabin",
    *(f"Emb_{p}" for p in PORTS),
    "GroupKnown", "GroupSurvRate",
)
GROUP_FEATURES = ("GroupKnown", "GroupSurvRate")

RAW_COLUMNS = ("PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked")


@dataclass
class FeatureState:
    """Статистики из обучающих данных. Сохраняются вместе с моделью."""

    age_median_by_title: dict[str, float]
    age_median: float
    fare_pp_median_by_class: dict[str, float]
    survival_prior: float
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> FeatureState:
        return cls(**json.loads(s))


@dataclass
class Context:
    """Кого ещё мы знаем на момент предсказания.

    passengers: все известные пассажиры (без исходов) — для размеров групп по билету.
    outcomes:   пассажиры с известным исходом (PassengerId, Ticket, Survived) — для «судьбы попутчиков».
    """

    passengers: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema={"PassengerId": pl.Int64, "Ticket": pl.String}))
    outcomes: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema={"PassengerId": pl.Int64, "Ticket": pl.String, "Survived": pl.Int64}))


def _base(df: pl.DataFrame) -> pl.DataFrame:
    """Признаки, не требующие статистик: титул, семья, флаги."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"нет колонок: {missing}")
    return df.select(RAW_COLUMNS).with_columns(
        pl.col("Name").str.extract(r",\s*([^\.]+)\.").replace_strict(TITLE_MAP, default="Rare").alias("Title"),
        (pl.col("Sex") == "male").cast(pl.Float32).alias("IsMale"),
        (pl.col("SibSp").fill_null(0) + pl.col("Parch").fill_null(0) + 1).alias("FamilySize"),
        pl.col("Age").is_null().cast(pl.Float32).alias("AgeMissing"),
        pl.col("Cabin").is_not_null().cast(pl.Float32).alias("HasCabin"),
        pl.col("Embarked").fill_null("S").alias("Embarked"),
    )


def _group_sizes(df: pl.DataFrame, ctx: Context) -> pl.DataFrame:
    everyone = pl.concat([df.select("PassengerId", "Ticket"), ctx.passengers.select("PassengerId", "Ticket")]).unique("PassengerId")
    sizes = everyone.group_by("Ticket").agg(pl.len().cast(pl.Float32).alias("TicketGroupSize"))
    return df.join(sizes, on="Ticket", how="left")


def fit(train: pl.DataFrame, ctx: Context | None = None) -> FeatureState:
    """Считает статистики заполнения по обучающим данным (должны содержать Survived)."""
    ctx = ctx or Context(passengers=train.select("PassengerId", "Ticket"))
    b = _group_sizes(_base(train), ctx).with_columns((pl.col("Fare") / pl.col("TicketGroupSize")).alias("FarePP"))
    age = b.drop_nulls("Age")
    by_title = {r["Title"]: float(r["Age"]) for r in age.group_by("Title").agg(pl.col("Age").median()).iter_rows(named=True)}
    by_class = {str(r["Pclass"]): float(r["FarePP"]) for r in b.drop_nulls("FarePP").group_by("Pclass").agg(pl.col("FarePP").median()).iter_rows(named=True)}
    return FeatureState(
        age_median_by_title=by_title,
        age_median=float(age["Age"].median()),
        fare_pp_median_by_class=by_class,
        survival_prior=float(train["Survived"].mean()),
    )


def transform(df: pl.DataFrame, state: FeatureState, ctx: Context | None = None) -> pl.DataFrame:
    """Сырые колонки → матрица признаков `FEATURES` (float32) + PassengerId.

    Порядок колонок фиксирован: это контракт с артефактом модели.
    """
    ctx = ctx or Context()
    b = _group_sizes(_base(df), ctx)
    age_fill = pl.col("Title").replace_strict(state.age_median_by_title, default=state.age_median, return_dtype=pl.Float64)
    fare_fill = pl.col("Pclass").cast(pl.String).replace_strict(state.fare_pp_median_by_class, default=None, return_dtype=pl.Float64)
    b = b.with_columns(
        pl.col("Age").fill_null(age_fill).alias("Age"),
        (pl.col("Fare") / pl.col("TicketGroupSize")).fill_null(fare_fill).alias("FarePP"),
    ).with_columns(
        ((pl.col("Age") < CHILD_AGE) | (pl.col("Title") == "Master")).cast(pl.Float32).alias("IsChild"),
        (pl.col("FamilySize") == 1).cast(pl.Float32).alias("IsAlone"),
        (pl.col("FamilySize") >= LARGE_FAMILY).cast(pl.Float32).alias("LargeFamily"),
        pl.col("FarePP").log1p().alias("LogFarePerPerson"),
        *(pl.col("Title").eq(t).cast(pl.Float32).alias(f"Title_{t}") for t in TITLES),
        *(pl.col("Embarked").eq(p).cast(pl.Float32).alias(f"Emb_{p}") for p in PORTS),
    )

    # Судьба попутчиков: другие пассажиры с тем же билетом и известным исходом (себя исключаем).
    mates = (b.select("PassengerId", "Ticket")
             .join(ctx.outcomes.select("Ticket", pl.col("PassengerId").alias("_mate"), "Survived"), on="Ticket", how="inner")
             .filter(pl.col("_mate") != pl.col("PassengerId"))
             .group_by("PassengerId")
             .agg(pl.len().cast(pl.Float32).alias("GroupKnown"), pl.col("Survived").mean().alias("GroupSurvRate")))
    b = b.join(mates, on="PassengerId", how="left").with_columns(
        pl.col("GroupKnown").fill_null(0.0),
        pl.col("GroupSurvRate").fill_null(state.survival_prior),
    )
    return b.select("PassengerId", *(pl.col(c).cast(pl.Float32) for c in FEATURES))


def to_numpy(features: pl.DataFrame):
    return features.select(FEATURES).to_numpy().astype("float32")


def full_context(train: pl.DataFrame, test: pl.DataFrame | None = None) -> Context:
    """Прод-контекст: знаем всех пассажиров рейса и все исходы архива."""
    pas = [train.select("PassengerId", "Ticket")] + ([test.select("PassengerId", "Ticket")] if test is not None else [])
    return Context(passengers=pl.concat(pas), outcomes=train.select("PassengerId", "Ticket", pl.col("Survived").cast(pl.Int64)))


def fold_context(train_fold: pl.DataFrame, all_passengers: pl.DataFrame) -> Context:
    """CV-контекст: размеры групп по всем (без исходов), исходы — только обучающего фолда."""
    return Context(passengers=all_passengers.select("PassengerId", "Ticket"),
                   outcomes=train_fold.select("PassengerId", "Ticket", pl.col("Survived").cast(pl.Int64)))


