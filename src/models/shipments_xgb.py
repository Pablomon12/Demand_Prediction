from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib
from xgboost import XGBRegressor
from src.evaluation.metrics import mae, rmse, mape
@dataclass
class ShipmentsConfig:
    n_estimators: int = 800
    learning_rate: float = 0.02
    max_depth: int = 4
    random_state: int = 42

def train_shipments_model(
    X_train,
    y_train,
    X_test,
    y_test,
    cfg: ShipmentsConfig,
    model_out_path: str = "reports/shipments_model.joblib",
):
    """
    Entrena un modelo de XGBoost para predecir el delivery_count
    """
    if "pred_order_volume" not in X_train.columns:
        raise ValueError("Falta la feature 'pred_order_volume' en X_train")
    if "pred_order_volume" not in X_test.columns:
        raise ValueError("Falta la feature 'pred_order_volume' en X_test")
    
    # Baseline simple: delivery ~= k * pred_order_volume
    eps = 1e-8
    k = float(np.mean(y_train / np.maximum(X_train["pred_order_volume"].values, eps)))
    baseline_pred = k * X_test["pred_order_volume"].values
    baseline_metrics_shipments = {
        "mae": mae(y_test, baseline_pred),
        "rmse": rmse(y_test, baseline_pred),
        "mape": mape(y_test, baseline_pred),
        "k": k,
    }
    model = XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        objective="reg:squarederror",
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    xgb_metrics_shipments = {
        "mae": mae(y_test, pred),
        "rmse": rmse(y_test, pred),
        "mape": mape(y_test, pred),
    }

    # Guardar modelo de envíos
    out = Path(model_out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    return {
        "model": model,
        "baseline_metrics": baseline_metrics_shipments,
        "xgb_metrics": xgb_metrics_shipments,
        "pred_test": pred,
        "pred_train": model.predict(X_train),
    }