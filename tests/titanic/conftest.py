from pathlib import Path

import polars as pl
import pytest

DATA = Path(__file__).resolve().parents[2] / "data" / "kaggle" / "titanic"


def _mk(rows):
    cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
    return pl.DataFrame(rows, schema=dict(zip(cols, [pl.Int64, pl.Int64, pl.Int64, pl.String, pl.String, pl.Float64,
                                                     pl.Int64, pl.Int64, pl.String, pl.Float64, pl.String, pl.String])), orient="row")


@pytest.fixture
def tiny():
    """Маленький синтетический архив: две семьи и одиночки."""
    return _mk([
        (1, 0, 3, "Smith, Mr. John", "male", 40.0, 1, 2, "T1", 30.0, None, "S"),
        (2, 0, 3, "Smith, Mrs. Anna", "female", 38.0, 1, 2, "T1", 30.0, None, "S"),
        (3, 0, 3, "Smith, Master. Tom", "male", None, 0, 2, "T1", 30.0, None, "S"),
        (4, 1, 1, "Lee, Mrs. Kate", "female", 50.0, 0, 1, "T2", 100.0, "B12", "C"),
        (5, 1, 1, "Lee, Miss. Sue", "female", 20.0, 0, 1, "T2", 100.0, "B12", "C"),
        (6, 0, 2, "Doe, Mr. Bob", "male", 30.0, 0, 0, "T3", 13.0, None, "Q"),
        (7, 1, 2, "Roe, Miss. Ann", "female", None, 0, 0, "T4", 13.0, None, None),
        (8, 0, 3, "Poe, Rev. Ed", "male", 60.0, 0, 0, "T5", 8.0, None, "S"),
    ])


@pytest.fixture(scope="session")
def kaggle_data():
    if not (DATA / "train.csv").exists():
        pytest.skip("нет данных: make download C=titanic")
    return pl.read_csv(DATA / "train.csv"), pl.read_csv(DATA / "test.csv")
