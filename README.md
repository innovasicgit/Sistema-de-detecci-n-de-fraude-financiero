# Streamlit ML Model Deployment

Base de una aplicación Streamlit para desplegar modelos de machine learning usando arquitectura en capas.

## Arquitectura

```text
app.py                         Punto de entrada
src/
  presentation/                UI y eventos de Streamlit
  application/                 Casos de uso y orquestación
  domain/                      Entidades y contratos del negocio
  infrastructure/              Carga de modelos, archivos y servicios externos
  config/                      Configuración de la aplicación
tests/                         Pruebas automatizadas
artifacts/                     Modelos serializados (no versionados)
```

La dependencia debe fluir hacia adentro: presentación -> aplicación -> dominio. La infraestructura implementa los contratos definidos por el dominio, por lo que conectar un modelo real no obliga a modificar la interfaz de Streamlit ni el caso de uso.

El flujo batch actual usa `AnalyzeDatasetUseCase` para transformar un `DataFrame` en un `AnalysisResult` con registros clasificados, métricas y distribución. `DemoBatchModelLoader` genera predicciones temporales para validar la interfaz; posteriormente se reemplazará por un adaptador que cargue los modelos entrenados.

## Ejecutar

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

El flujo actual usa `DemoModelLoader` como sustituto. Para conectar un modelo real, implementa `ModelLoader` en `src/infrastructure/ml/` y cambia la composición en `src/presentation/streamlit_app.py`.

## Pruebas

```powershell
python -m pytest
```
