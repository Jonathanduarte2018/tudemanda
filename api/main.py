from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from water_pred.forecast import forecast as run_forecast

app = FastAPI(title="Water Consumption Forecast API")


class ForecastRequest(BaseModel):
    model_path: str = Field(..., description="Path to saved model .joblib")
    steps: int = Field(..., ge=1, le=365)
    exog: Optional[list[dict]] = Field(None, description="Optional future exogenous rows")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast")
async def forecast(req: ForecastRequest) -> dict:
    model_path = Path(req.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Model path not found")

    exog_df: Optional[pd.DataFrame] = None
    if req.exog is not None:
        exog_df = pd.DataFrame(req.exog)

    try:
        yhat = run_forecast(str(model_path), steps=req.steps, exog_future=exog_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"predictions": yhat.tolist()}
