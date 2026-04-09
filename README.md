## Prediccion de Demanda y Envios

### Objetivo
Este proyecto se centra exclusivamente en modelado predictivo para:

- **Demanda** (`order_volume`)
- **Numero de envios** (`delivery_count`)

La implementacion actual entrena:

1. un modelo de demanda (baseline + XGBoost),
2. un modelo de envios apoyado en la prediccion de demanda.

### Dataset
El dataset utilizado esta en `src/data/logistics_dataset_2020_2024.csv` y contiene series temporales agregadas por fecha.

### Estructura del proyecto
- `src/data/`: carga de datos, features y split temporal.
- `src/models/`: modelos de demanda y envios.
- `src/evaluation/`: metricas (MAE, RMSE, MAPE).
- `src/pipelines/train_pipeline.py`: entrenamiento end-to-end de demanda + envios.
- `reports/`: modelos entrenados (`.joblib`).

### Instalacion
```bash
python -m pip install -r requirements.txt
```

### Ejecucion
```bash
python -m src.pipelines.train_pipeline --csv-path src/data/logistics_dataset_2020_2024.csv
```

### Roadmap
En una siguiente fase se anadira un dataset con **SKUs** y **localizaciones de almacen** para extender el proyecto hacia:

- optimizacion de rutas internas de picking,
- y redistribucion de medicamentos en el almacen.# Demand_Prediction
