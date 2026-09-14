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
py -3.10 -m venv .venv310
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv310\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Si PowerShell bloquea la ejecución de scripts, puedes ejecutar la aplicación sin activar el entorno:

```powershell
.\.venv310\Scripts\python.exe -m streamlit run app.py
```

Para ejecutar las pruebas sin activar el entorno:

```powershell
.\.venv310\Scripts\python.exe -m pytest
```

El proyecto utiliza los modelos entrenados almacenados en `src/models/`. Si los modelos se descargan mediante Git LFS, ejecuta `git lfs pull` después de clonar el repositorio.

## Pruebas

```powershell
python -m pytest
```
