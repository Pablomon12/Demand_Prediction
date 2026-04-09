import argparse

from src.data.loaders import load_logistics_dataset
from src.data.features import ensure_time_and_lags, get_feature_columns
from src.data.splitting import time_split_xy
from src.models.demand_xgb import DemandXGBConfig, train_demand_model
from src.models.shipments_xgb import ShipmentsConfig, train_shipments_model

def main():
    parser = argparse.ArgumentParser(description="Entrena modelo de demanda (order_volume)")
    parser.add_argument("--csv-path", type=str, default="src/data/logistics_dataset_2020_2024.csv")
    parser.add_argument("--cutoff-date", type=str, default="2023-10-01")
    parser.add_argument("--model-out-path", type=str, default="reports/demand_model.joblib")
    parser.add_argument("--model-out-path-shipments", type=str, default="reports/shipments_model.joblib")
    args = parser.parse_args()

    # 1) Cargar y preparar
    df = load_logistics_dataset(args.csv_path)
    df = ensure_time_and_lags(df, target_col="order_volume")
    features = get_feature_columns(df)

    # 2) Split temporal
    X_train, X_test, y_train, y_test = time_split_xy(
        df=df,
        cutoff_date=args.cutoff_date,
        target_col="order_volume",
        feature_cols=features,
        dropna=True,
    )

    # 3) Train
    cfg = DemandXGBConfig()
    demand_model, baseline_metrics, xgb_metrics = train_demand_model(
        X_train, y_train, X_test, y_test, cfg, model_out_path=args.model_out_path
    )

    print("=== RESULTADOS DEMANDA ===")
    print("Baseline (lag_1_order):", baseline_metrics)
    print("XGBoost:", xgb_metrics)

    # Predicciones de demanda para alimentar el modelo de envíos
    pred_order_train = demand_model.predict(X_train)
    pred_order_test = demand_model.predict(X_test)

    # 4) Train Shipments Model
    df = load_logistics_dataset(args.csv_path)
    df = ensure_time_and_lags(df, target_col="delivery_count")
    features = get_feature_columns(df)
    X_train_ship, X_test_ship, y_train_ship, y_test_ship = time_split_xy(
        df=df,
        cutoff_date=args.cutoff_date,
        target_col="delivery_count",
        feature_cols=features,
        dropna=True,
    )
    X_train_ship = X_train_ship.copy()
    X_test_ship = X_test_ship.copy()
    X_train_ship["pred_order_volume"] = pred_order_train
    X_test_ship["pred_order_volume"] = pred_order_test

    cfg = ShipmentsConfig()
    result_shipments = train_shipments_model(
        X_train_ship,
        y_train_ship,
        X_test_ship,
        y_test_ship,
        cfg,
        model_out_path=args.model_out_path_shipments,
    )
    baseline_metrics_shipments = result_shipments["baseline_metrics"]
    xgb_metrics_shipments = result_shipments["xgb_metrics"]

    print("=== RESULTADOS ENVIOS ===")
    print("Baseline:", baseline_metrics_shipments)
    print("XGBoost:", xgb_metrics_shipments)


if __name__ == "__main__":
    main()