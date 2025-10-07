from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class SarimaxConfig:
    order: Tuple[int, int, int] = (1, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False


def fit_sarimax(
    y: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    config: Optional[SarimaxConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = SarimaxConfig()

    model = SARIMAX(
        y,
        exog=exog,
        order=config.order,
        seasonal_order=config.seasonal_order,
        enforce_stationarity=config.enforce_stationarity,
        enforce_invertibility=config.enforce_invertibility,
        trend="n",
    )
    res = model.fit(disp=False)

    return {
        "params": {
            "order": config.order,
            "seasonal_order": config.seasonal_order,
            "enforce_stationarity": config.enforce_stationarity,
            "enforce_invertibility": config.enforce_invertibility,
        },
        "fit_result": res,
    }


def grid_search_sarimax(
    y: pd.Series,
    orders: Tuple[Tuple[int, int, int], ...],
    seasonal_orders: Tuple[Tuple[int, int, int, int], ...] = ((0, 0, 0, 0),),
    exog: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    best_ic = np.inf
    best: Optional[Dict[str, Any]] = None

    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                result = fit_sarimax(
                    y=y,
                    exog=exog,
                    config=SarimaxConfig(order=order, seasonal_order=seasonal_order),
                )
                ic = result["fit_result"].aic
                if ic < best_ic:
                    best_ic = ic
                    best = result
            except Exception:
                continue

    if best is None:
        raise RuntimeError("All SARIMAX configurations failed to fit")

    return best


def save_model(obj: Dict[str, Any], path: str) -> None:
    # Replace statsmodels results with params dict and pickled result
    fit_result = obj.pop("fit_result")
    joblib.dump({"metadata": obj["params"], "fit_result": fit_result}, path)


def load_model(path: str) -> Dict[str, Any]:
    return joblib.load(path)
