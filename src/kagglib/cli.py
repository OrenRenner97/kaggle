"""CLI: uv run kagglib <соревнование> <команда> ...

  kagglib titanic train   --data data/kaggle/titanic --out models/titanic
  kagglib titanic predict --model models/titanic --input data/kaggle/titanic/test.csv --output sub.csv [--backend onnx|native|logreg]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _titanic(args) -> None:
    import polars as pl

    if args.cmd == "train":
        from kagglib.titanic import model

        tr = pl.read_csv(Path(args.data) / "train.csv")
        te_path = Path(args.data) / "test.csv"
        te = pl.read_csv(te_path) if te_path.exists() else None
        t = model.train(tr, extra_passengers=te)
        out = model.export(t, Path(args.out), golden_df=te.head(50) if te is not None else None,
                           build_native=not args.no_native)
        print(f"пакет артефактов: {out}")
        for p in sorted(out.iterdir()):
            print(f"  {p.name:<26} {p.stat().st_size:>10,} B")
    elif args.cmd == "predict":
        from kagglib.titanic.predict import Predictor

        pred = Predictor(Path(args.model), backend=args.backend)
        res = pred.predict(pl.read_csv(args.input))
        res.select("PassengerId", "Survived").write_csv(args.output)
        print(f"{res.height} предсказаний ({args.backend}) → {args.output}; доля 'выжил' = {res['Survived'].mean():.3f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kagglib")
    sub = ap.add_subparsers(dest="competition", required=True)

    t = sub.add_parser("titanic").add_subparsers(dest="cmd", required=True)
    tr = t.add_parser("train", help="обучить и собрать пакет артефактов")
    tr.add_argument("--data", default="data/kaggle/titanic")
    tr.add_argument("--out", default="models/titanic")
    tr.add_argument("--no-native", action="store_true", help="не собирать libtitanic.so")
    pr = t.add_parser("predict", help="предсказать по пакету артефактов")
    pr.add_argument("--model", default="models/titanic")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--backend", default="onnx", choices=["onnx", "native", "logreg"])

    args = ap.parse_args(argv)
    {"titanic": _titanic}[args.competition](args)


if __name__ == "__main__":
    main()
