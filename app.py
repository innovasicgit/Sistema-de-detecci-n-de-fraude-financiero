# LightGBM se importa antes de pandas/Streamlit para inicializar correctamente
# su biblioteca nativa en Windows antes de deserializar los modelos.
import lightgbm

from src.presentation.streamlit_app import run_app


if __name__ == "__main__":
    # El punto de entrada solo delega la composición y el renderizado en la capa de presentación.
    run_app()
