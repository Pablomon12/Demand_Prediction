from __future__ import annotations

from pathlib import Path

from src.data.features import ensure_time_and_lags, get_feature_columns
from src.data.loaders import load_logistics_dataset
from src.data.splitting import time_split_xy
from src.models.demand_xgb import DemandXGBConfig, train_demand_model
from src.models.shipments_xgb import ShipmentsConfig, train_shipments_model


def main() -> None:
    csv_path = "src/data/logistics_dataset_2020_2024.csv"
    cutoff_date = "2023-10-01"

    print("=== SMOKE TEST: DEMANDA + ENVIOS ===")
    print(f"CSV: {csv_path}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"No existe el dataset en: {csv_path}")

    # 1) Carga y preprocesado base para demanda
    df_demand = load_logistics_dataset(csv_path)
    df_demand = ensure_time_and_lags(df_demand, target_col="order_volume")
    demand_features = get_feature_columns(df_demand)
    if not demand_features:
        raise RuntimeError("No se detectaron features para demanda.")

    X_train, X_test, y_train, y_test = time_split_xy(
        df=df_demand,
        cutoff_date=cutoff_date,
        target_col="order_volume",
        feature_cols=demand_features,
        dropna=True,
    )

    assert len(X_train) > 0 and len(X_test) > 0, "Split de demanda vacio."
    assert X_train.shape[1] == len(demand_features), "N de features inconsistente en demanda."
    print(f"Demanda split OK: X_train={X_train.shape}, X_test={X_test.shape}")

    # 2) Entrenamiento demanda
    demand_cfg = DemandXGBConfig()
    demand_model, demand_baseline, demand_xgb = train_demand_model(
        X_train,
        y_train,
        X_test,
        y_test,
        demand_cfg,
        model_out_path="reports/demand_model.joblib",
    )

    for k in ("mae", "rmse", "mape"):
        assert k in demand_baseline and k in demand_xgb, f"Falta metrica {k} en demanda."
    print("Demanda train OK.")
    print("Baseline demanda:", demand_baseline)
    print("XGBoost demanda:", demand_xgb)

    # 3) Preparacion y entrenamiento envios
    pred_order_train = demand_model.predict(X_train)
    pred_order_test = demand_model.predict(X_test)

    df_ship = load_logistics_dataset(csv_path)
    df_ship = ensure_time_and_lags(df_ship, target_col="delivery_count")
    ship_features = get_feature_columns(df_ship)

    X_train_ship, X_test_ship, y_train_ship, y_test_ship = time_split_xy(
        df=df_ship,
        cutoff_date=cutoff_date,
        target_col="delivery_count",
        feature_cols=ship_features,
        dropna=True,
    )
    X_train_ship = X_train_ship.copy()
    X_test_ship = X_test_ship.copy()
    X_train_ship["pred_order_volume"] = pred_order_train
    X_test_ship["pred_order_volume"] = pred_order_test

    ship_cfg = ShipmentsConfig()
    ship_res = train_shipments_model(
        X_train_ship,
        y_train_ship,
        X_test_ship,
        y_test_ship,
        ship_cfg,
        model_out_path="reports/shipments_model.joblib",
    )

    for k in ("mae", "rmse", "mape", "k"):
        assert k in ship_res["baseline_metrics"], f"Falta metrica {k} en baseline de envios."
    for k in ("mae", "rmse", "mape"):
        assert k in ship_res["xgb_metrics"], f"Falta metrica {k} en xgb de envios."

    assert len(ship_res["pred_test"]) == len(y_test_ship), "Predicciones de envios con longitud incorrecta."
    print("Envios train OK.")
    print("Baseline envios:", ship_res["baseline_metrics"])
    print("XGBoost envios:", ship_res["xgb_metrics"])

    # 4) Comprobacion de artefactos
    assert Path("reports/demand_model.joblib").exists(), "No se guardo demand_model.joblib"
    assert Path("reports/shipments_model.joblib").exists(), "No se guardo shipments_model.joblib"
    print("Artefactos OK: reports/demand_model.joblib, reports/shipments_model.joblib")

    print("\nSMOKE TEST COMPLETADO SIN ERRORES.")


if __name__ == "__main__":
    main()

