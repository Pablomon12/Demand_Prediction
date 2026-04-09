from dataclasses import dataclass
from pathlib import Path
import joblib
from xgboost import XGBRegressor

from src.evaluation.metrics import mae, rmse, mape


@dataclass
class DemandXGBConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.01
    max_depth: int = 5
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int = 42


def train_demand_model(
    X_train,
    y_train,
    X_test,
    y_test,
    cfg: DemandXGBConfig,
    model_out_path: str = "reports/demand_model.joblib",
):
    # 1) Baseline naive: usar lag_1_order
    if "lag_1_order" not in X_test.columns:
        raise ValueError("La feature 'lag_1_order' es necesaria para la baseline.")
    baseline_pred = X_test["lag_1_order"].values

    baseline_metrics = {
        "mae": mae(y_test, baseline_pred),
        "rmse": rmse(y_test, baseline_pred),
        "mape": mape(y_test, baseline_pred),
    }

    # 2) Modelo XGBoost
    model = XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.random_state,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)

    xgb_metrics = {
        "mae": mae(y_test, pred),
        "rmse": rmse(y_test, pred),
        "mape": mape(y_test, pred),
    }

    # 3) Guardar modelo
    out = Path(model_out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    return model, baseline_metrics, xgb_metrics