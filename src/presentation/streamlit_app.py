import streamlit as st

from src.application.use_cases.analyze_dataset import AnalyzeDatasetUseCase
from src.infrastructure.ml.model_repository import ModelRepository
from src.presentation.components import (
    load_styles,
    render_brand,
    render_analysis,
    render_config,
    render_guide,
    render_home,
    render_evaluation,
)


def run_app() -> None:
    """Configura Streamlit y conecta cada pantalla con sus dependencias."""
    # Streamlit reejecuta esta función en cada interacción; los modelos se obtienen
    # desde un repositorio cacheado y el estado persistente vive en session_state.
    st.set_page_config(page_title="Fraude Contable", page_icon=":bar_chart:", layout="wide")
    appearance = st.session_state.get("appearance", "Claro")
    load_styles(appearance)
    page = st.session_state.get("page", "Inicio")

    with st.sidebar:
        # La barra lateral concentra navegación, apariencia y la entrada al modo avanzado.
        render_brand()
        st.divider()
        navigation = [
            ("Inicio", ":material/home:"),
            ("Guía de uso", ":material/menu_book:"),
            ("Analizar archivo", ":material/manage_search:"),
            ("Configuración", ":material/settings:"),
        ]
        selected_page = page
        for label, icon in navigation:
            if st.button(
                label,
                icon=icon,
                key=f"navigation_{label}",
                use_container_width=True,
                type="primary" if page == label else "secondary",
            ):
                selected_page = label
                st.session_state.page = label
                st.rerun()
        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
        st.divider()
        dark_mode = st.toggle("Modo oscuro", value=appearance == "Oscuro")
        requested_appearance = "Oscuro" if dark_mode else "Claro"
        if requested_appearance != appearance:
            st.session_state.appearance = requested_appearance
            st.rerun()
        if st.button(
            "Evaluar modelo",
            icon=":material/assessment:",
            use_container_width=True,
            type="primary" if page == "Evaluar modelo" else "secondary",
            key="navigation_Evaluar modelo",
        ):
            st.session_state.page = "Evaluar modelo"
            st.rerun()
        st.caption("Versión 1.0 · Prototipo")

    # Cada vista delega su lógica en un componente de presentación especializado.
    if selected_page == "Inicio":
        render_home()
    elif selected_page == "Guía de uso":
        render_guide(ModelRepository())
    elif selected_page == "Analizar archivo":
        render_analysis(AnalyzeDatasetUseCase(ModelRepository()))
    elif selected_page == "Evaluar modelo":
        render_evaluation(AnalyzeDatasetUseCase(ModelRepository()))
    else:
        render_config()
