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
)


def run_app() -> None:
    st.set_page_config(page_title="Fraude Contable", page_icon=":bar_chart:", layout="wide")
    appearance = st.session_state.get("appearance", "Claro")
    load_styles(appearance)
    page = st.session_state.get("page", "Inicio")

    with st.sidebar:
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
        st.caption("Versión 1.0 · Prototipo")

    if selected_page == "Inicio":
        render_home()
    elif selected_page == "Guía de uso":
        render_guide(ModelRepository())
    elif selected_page == "Analizar archivo":
        render_analysis(AnalyzeDatasetUseCase(ModelRepository()))
    else:
        render_config()
