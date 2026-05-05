# Predicción de suscripción de depósito bancario

Práctica de Aprendizaje Automático para predecir si un cliente suscribirá un depósito bancario. El repositorio incluye el análisis exploratorio, la comparación de modelos, el modelo final entrenado, las predicciones sobre el conjunto de competición y una aplicación Streamlit para probar inferencias de forma interactiva.

Autores:

- Diego Valladares
- Luis del Valle

## Contenido

- `exercise.ipynb`: análisis exploratorio, preparación de datos, selección de modelos, ajuste de hiperparámetros y evaluación final.
- `modelo_final_predicciones.ipynb`: carga del modelo final y generación de predicciones para el conjunto de competición.
- `mystreamlit.py`: aplicación Streamlit para inferencia.
- `modelo_final.joblib`: modelo final serializado.
- `predicciones.csv`: predicciones generadas para el conjunto de competición.
- `bank_05.pkl` y `bank_competition.pkl`: datasets usados en la práctica.
- `evidencias_streamlit.pdf`: evidencias de la aplicación Streamlit.

## Instalación

Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar la aplicación

```bash
streamlit run mystreamlit.py
```

La aplicación carga `modelo_final.joblib`, permite introducir las variables de un cliente y muestra la predicción junto con las probabilidades estimadas.

## Flujo de trabajo

1. Ejecutar `exercise.ipynb` para reproducir el análisis, la selección de modelos y el entrenamiento del modelo final.
2. Ejecutar `modelo_final_predicciones.ipynb` para cargar `modelo_final.joblib` y generar `predicciones.csv`.
3. Ejecutar `mystreamlit.py` para probar la aplicación de inferencia.

## Reproducibilidad

Las dependencias principales están en `requirements.txt`. La práctica fija una semilla basada en el NIA para que los resultados sean reproducibles.

## Nota sobre los datos

La licencia MIT de este repositorio aplica al código. Los datasets incluidos se distribuyen como parte de la práctica y mantienen las condiciones de uso de la asignatura o de su fuente original.
