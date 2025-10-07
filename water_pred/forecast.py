from __future__ import annotations

from typing import Optional

import pandas as pd

from .model import load_model


def forecast(
    model_path: str,
    steps: int,
    exog_future: Optional[pd.DataFrame] = None,
) -> pd.Series:
    obj = load_model(model_path)
    fit = obj["fit_result"]
    pred = fit.get_forecast(steps=steps, exog=exog_future)
    return pred.predicted_mean
