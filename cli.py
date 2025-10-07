from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_pred.data import load_time_series
from water_pred.model import grid_search_sarimax, save_model
from water_pred.forecast import forecast as run_forecast


def cmd_train(args: argparse.Namespace) -> None:
    series = load_time_series(
        csv_path=args.csv,
        date_col=args.date_col,
        value_col=args.value_col,
        location_col=args.location_col,
        location_value=args.location_value,
    )

    orders = tuple((p, d, q) for p in range(0, 3) for d in range(0, 2) for q in range(0, 3))
    seasonal_orders = tuple((P, D, Q, args.season_length) for P in range(0, 2) for D in range(0, 2) for Q in range(0, 2)) if args.season_length else ((0, 0, 0, 0),)

    best = grid_search_sarimax(series, orders=orders, seasonal_orders=seasonal_orders)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_model(best, args.output)
    print(f"Model saved to {args.output}")


def cmd_forecast(args: argparse.Namespace) -> None:
    exog = None
    if args.exog_csv:
        exog_df = pd.read_csv(args.exog_csv)
        exog_df = exog_df.drop(columns=[args.date_col], errors="ignore")
        exog = exog_df

    pred = run_forecast(args.model, steps=args.steps, exog_future=exog)
    print(pred.to_string())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Water consumption forecasting")
    sub = p.add_subparsers(required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--csv", required=True)
    pt.add_argument("--date-col", default="date")
    pt.add_argument("--value-col", default="consumption")
    pt.add_argument("--location-col")
    pt.add_argument("--location-value")
    pt.add_argument("--season-length", type=int)
    pt.add_argument("--output", default="models/water_sarimax.joblib")
    pt.set_defaults(func=cmd_train)

    pf = sub.add_parser("forecast")
    pf.add_argument("--model", required=True)
    pf.add_argument("--steps", type=int, required=True)
    pf.add_argument("--exog-csv")
    pf.add_argument("--date-col", default="date")
    pf.set_defaults(func=cmd_forecast)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
