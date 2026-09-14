import csv
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from src.application.use_cases.analyze_dataset import AnalyzeDatasetInput
from src.domain.entities.analysis import FRAUD_LABELS


ASSETS_DIR = Path(__file__).parent / "assets"


def load_styles(appearance: str = "Claro") -> None:
    """Carga el CSS base y aplica las variables del tema seleccionado."""
    css = (ASSETS_DIR / "styles.css").read_text(encoding="utf-8")
    theme_css = ""
    if appearance == "Oscuro":
        theme_css = """
        <style>
        :root {
          --bg: #121815; --surface: #1a211d; --surface-2: #212925;
          --border: #2b342f; --text: #edefed; --text-soft: #a6b0aa;
          --text-faint: #79837d; --green: #2e9c6d; --green-strong: #1e7a54;
          --green-soft: #1b2c24; --green-mid: #3fb57f; --alert: #e07059;
          --alert-soft: #2e1e1b; --ok: #5fcb8c; --ok-soft: #17281f;
          --shadow: 0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.35);
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--bg); }
        [data-testid="stDataFrame"] { background: var(--surface); }
        </style>
        """
    elif appearance == "Sistema":
        theme_css = """
        <style>
        @media (prefers-color-scheme: dark) {
          :root {
            --bg: #121815; --surface: #1a211d; --surface-2: #212925;
            --border: #2b342f; --text: #edefed; --text-soft: #a6b0aa;
            --text-faint: #79837d; --green: #2e9c6d; --green-strong: #1e7a54;
            --green-soft: #1b2c24; --green-mid: #3fb57f; --alert: #e07059;
            --alert-soft: #2e1e1b; --ok: #5fcb8c; --ok-soft: #17281f;
          }
          [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--bg); }
        }
        </style>
        """
    st.html(f"<style>{css}</style>{theme_css}")


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## Sistema de Detección")
        st.caption("de Fraude Contable")
        st.divider()
        page = st.radio(
            "Navegación",
            ["Inicio", "Guía de uso", "Analizar archivo", "Configuración"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Versión 1.0 · Prototipo")
    return page


def render_brand() -> None:
        st.markdown(
                """
                <div class="brand-block">
                    <div class="brand-mark">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path d="M4 19h16"/><path d="M6 19V9l4-4 4 4v10"/><path d="M14 19v-6l4-3v9"/>
                        </svg>
                    </div>
                    <div class="brand-name">Sistema de Detección<br>de Fraude Contable</div>
                </div>
                """,
                unsafe_allow_html=True,
        )


def render_home() -> None:
    """Renderiza la pantalla inicial y el resumen del último análisis."""
    st.markdown(
        """
        <div class="hero-card">
          <h1>Sistema de Detección de Posibles Fraudes Contables</h1>
          <p>Herramienta de apoyo para el análisis de información contable mediante modelos de inteligencia artificial.</p>
                    <div class="models-strip">
                        <span class="chip">Modelos disponibles: 4</span>
                        <span class="chip">LightGBM</span><span class="chip">Random Forest</span>
                        <span class="chip">XGBoost</span><span class="chip">Decision Tree</span>
                    </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Analizar archivo contable", type="primary"):
        st.session_state.page = "Analizar archivo"
        st.rerun()

    st.markdown('<div class="section-label">Resumen del último análisis</div>', unsafe_allow_html=True)
    metrics = st.columns(4)
    analysis = st.session_state.get("analysis")
    values = [
        ("Registros analizados", f"{analysis.total:,}" if analysis else "—"),
        ("Registros normales", f"{analysis.normal:,}" if analysis else "—"),
        ("Alertas detectadas (Posible fraude)", f"{analysis.alerts:,}" if analysis else "—"),
        ("Modelo utilizado", analysis.model_name if analysis else "Sin análisis"),
    ]
    for column, (label, value) in zip(metrics, values):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value muted">{value}</div></div>', unsafe_allow_html=True)


def render_guide(model_repository: Any) -> None:
    """Muestra instrucciones, variables del modelo y clases posibles."""
    st.title("Guía de uso")
    st.markdown('<p class="app-subtitle">Sigue estos pasos para analizar tu información contable, sin necesidad de conocimientos técnicos.</p>', unsafe_allow_html=True)
    steps = [("1", "Cargar CSV", "Sube el archivo contable en formato CSV."), ("2", "Validar información", "El sistema revisa las 19 variables requeridas."), ("3", "Ejecutar modelo", "Elige un modelo y ejecuta el análisis."), ("4", "Revisar resultados", "Consulta las clasificaciones y alertas."), ("5", "Descargar reporte", "Exporta los resultados para auditoría.")]
    columns = st.columns(5)
    for column, (number, title, description) in zip(columns, steps):
        with column:
            st.markdown(f'<div class="step-card"><div class="step-number">{number}</div><h4>{title}</h4><p>{description}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Variables requeridas</div>', unsafe_allow_html=True)
    with st.expander("Ver variables requeridas"):
        try:
            model = model_repository.load("LightGBM")
            st.write(list(model.feature_names_in_))
        except Exception:
            st.info("No fue posible cargar las variables del modelo en este momento.")

    st.markdown('<div class="section-label">Clasificaciones posibles</div>', unsafe_allow_html=True)
    classifications = [("Normal", "El registro no presenta patrones inusuales."), ("Pitufeo", "Fraccionamiento de montos para evitar controles."), ("Redondeo de cifras", "Patrones de redondeo poco habituales."), ("Fraude de nómina", "Irregularidades en pagos o empleados."), ("Pasivos como ingresos", "Reclasificación indebida de obligaciones."), ("Cuentas inexistentes", "Movimientos en cuentas no válidas.")]
    columns = st.columns(3)
    for column, (title, description) in zip(columns * 2, classifications):
        with column:
            st.markdown(f'<div class="classification-card"><h4>{title}</h4><p>{description}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">Las clasificaciones son alertas de apoyo y no reemplazan el juicio profesional del contador o auditor.</div>', unsafe_allow_html=True)


def render_analysis(analyze_use_case: Any) -> None:
    """Gestiona la predicción operativa sobre un CSV sin etiqueta real."""
    st.title("Análisis de fraude contable")
    st.markdown('<p class="app-subtitle">Cargue la información contable y seleccione el modelo que desea utilizar.</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">1. Cargar archivo</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Arrastre o seleccione un archivo CSV", type=["csv"], help="Tamaño máximo permitido: 500 MB")
    valid_file = False
    dataframe = None
    if uploaded_file is not None:
        try:
            file_signature = (
                uploaded_file.name,
                uploaded_file.size,
                getattr(uploaded_file, "file_id", None),
            )
            # Solo se relee el archivo cuando cambia su firma; así los reruns no
            # vuelven a parsear cientos de miles de filas.
            if st.session_state.get("dataframe_signature") != file_signature:
                uploaded_file.seek(0)
                dataframe = _read_csv(uploaded_file)
                st.session_state.dataframe = dataframe
                st.session_state.dataframe_signature = file_signature
                st.session_state.analysis = None
                st.session_state.results_page = 1
            else:
                dataframe = cast(pd.DataFrame, st.session_state.get("dataframe"))
            st.markdown(
                f'<div class="status-ok">Archivo cargado correctamente · {len(dataframe):,} registros · {len(dataframe.columns):,} columnas</div>',
                unsafe_allow_html=True,
            )
        except Exception as error:
            st.error(f"No fue posible leer el archivo: {error}")

    st.markdown('<div class="section-label">2. Seleccionar modelo</div>', unsafe_allow_html=True)
    model = st.selectbox("Modelo", ["LightGBM", "Random Forest", "XGBoost", "Decision Tree"], label_visibility="collapsed")
    st.caption("El modelo seleccionado analizará cada registro y asignará una clasificación.")
    if dataframe is not None:
        try:
            # Cada modelo puede declarar columnas distintas, por eso se valida
            # después de seleccionar el modelo y antes de habilitar el botón.
            validation = analyze_use_case.validate(dataframe, model)
            if validation.is_valid:
                st.markdown(
                    f'<div class="status-ok">✓ {len(validation.expected_features)} variables requeridas encontradas · {len(dataframe):,} registros disponibles</div>',
                    unsafe_allow_html=True,
                )
                valid_file = not dataframe.empty
                if dataframe.empty:
                    st.warning("El archivo CSV está vacío y no contiene registros para analizar.")
            else:
                missing = ", ".join(validation.missing_features)
                st.markdown(
                    f'<div class="status-alert">⚠ El archivo no contiene todas las variables requeridas. Faltan: {missing}</div>',
                    unsafe_allow_html=True,
                )
        except Exception as error:
            st.error(str(error))
    if st.button("Analizar archivo", type="primary", disabled=uploaded_file is None or not valid_file):
        dataframe = st.session_state.get("dataframe")
        if dataframe is not None:
            with st.spinner("Analizando registros..."):
                try:
                    # La inferencia ocurre únicamente tras una acción explícita del usuario.
                    st.session_state.analysis = analyze_use_case.execute(AnalyzeDatasetInput(dataframe, model))
                except ValueError as error:
                    st.error(str(error))
                except Exception:
                    st.error("No fue posible completar el análisis. Revise el archivo y el modelo seleccionado.")

    render_results()


def render_evaluation(analyze_use_case: Any) -> None:
    """Evalúa un modelo contra misstate y muestra reporte y matriz de confusión."""
    st.title("Evaluar modelo")
    st.markdown(
        '<p class="app-subtitle">Compare las predicciones con la etiqueta real del CSV.</p>',
        unsafe_allow_html=True,
    )
    st.info("Esta evaluación requiere una columna 'misstate' con valores de 0 a 5.")
    uploaded_file = st.file_uploader(
        "Cargue un CSV etiquetado", type=["csv"], key="evaluation_uploader"
    )
    model_name = st.selectbox(
        "Modelo a evaluar",
        ["LightGBM", "Random Forest", "XGBoost", "Decision Tree"],
        key="evaluation_model",
    )
    if uploaded_file is None:
        return

    try:
        signature = (uploaded_file.name, uploaded_file.size, getattr(uploaded_file, "file_id", None))
        if st.session_state.get("evaluation_signature") != signature:
            uploaded_file.seek(0)
            st.session_state.evaluation_dataframe = _read_csv(uploaded_file)
            st.session_state.evaluation_signature = signature
            st.session_state.evaluation_result = None
        dataframe = cast(pd.DataFrame, st.session_state.get("evaluation_dataframe"))
        if "misstate" not in dataframe.columns:
            st.error("El CSV no contiene la columna 'misstate'. No es posible calcular métricas reales.")
            return
        st.markdown(
            f'<div class="status-ok">Archivo etiquetado · {len(dataframe):,} registros · columna misstate encontrada</div>',
            unsafe_allow_html=True,
        )
        if st.button("Evaluar modelo", type="primary", key="run_evaluation"):
            with st.spinner("Calculando métricas y matriz de confusión..."):
                try:
                    # La evaluación se ejecuta al pulsar el botón y se conserva en session_state.
                    st.session_state.evaluation_result = analyze_use_case.evaluate(dataframe, model_name)
                except ValueError as error:
                    st.error(str(error))
        evaluation = st.session_state.get("evaluation_result")
        if evaluation is None:
            return

        st.subheader(f"Evaluación de {model_name}")
        evaluation_metrics = st.columns(2)
        evaluation_metrics[0].metric("Accuracy", f"{evaluation.report['accuracy']:.2%}")
        evaluation_metrics[1].metric(
            "F1 macro", f"{evaluation.report['macro avg']['f1-score']:.2%}"
        )
        # Se transforma el diccionario de sklearn en una tabla legible para la interfaz.
        report_rows = []
        for label in evaluation.labels:
            name = FRAUD_LABELS.get(label, str(label))
            values = evaluation.report[name]
            report_rows.append({
                "Clase": name,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-score": values["f1-score"],
                "Soporte": int(values["support"]),
            })
        for average in ("macro avg", "weighted avg"):
            values = evaluation.report[average]
            report_rows.append({
                "Clase": "Promedio macro" if average == "macro avg" else "Promedio ponderado",
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-score": values["f1-score"],
                "Soporte": int(values["support"]),
            })
        report_frame = pd.DataFrame(report_rows)
        for column in ("Precision", "Recall", "F1-score"):
            report_frame[column] = report_frame[column].map(lambda value: f"{value:.4f}")
        st.dataframe(report_frame, use_container_width=True, hide_index=True)

        st.markdown("### Matriz de confusión")
        labels = [FRAUD_LABELS.get(label, str(label)) for label in evaluation.labels]
        # Las filas son clases reales y las columnas son clases predichas.
        confusion = pd.DataFrame(evaluation.confusion_matrix, index=labels, columns=labels)
        confusion.index.name = "Real \\ Predicha"
        st.dataframe(confusion, use_container_width=True)
    except Exception as error:
        st.error(f"No fue posible leer el CSV de evaluación: {error}")


def _read_csv(uploaded_file: Any) -> pd.DataFrame:
    """Lee CSV regionales con el motor C, evitando el parser Python completo."""
    # Se inspeccionan solo 64 KB para detectar el separador; pandas procesa el resto.
    sample = uploaded_file.read(64 * 1024)
    uploaded_file.seek(0)
    if isinstance(sample, bytes):
        sample = sample.decode("utf-8-sig", errors="replace")
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    return pd.read_csv(uploaded_file, sep=delimiter, engine="c", low_memory=False)


def render_results() -> None:
    """Presenta métricas, distribución, filtros, detalle y descarga del análisis."""
    analysis = st.session_state.get("analysis")
    if analysis is None:
        return
    st.divider()
    st.markdown("## Resultados del análisis")
    st.caption(f"Modelo utilizado: {analysis.model_name}")

    metrics = st.columns(4)
    metric_values = [
        ("Total analizado", f"{analysis.total:,}", "metric-neutral"),
        ("Registros normales", f"{analysis.normal:,}", "metric-ok"),
        ("Alertas detectadas", f"{analysis.alerts:,}", "metric-alert"),
        ("Porcentaje de alertas", f"{analysis.alert_rate:.2f} %", "metric-neutral"),
    ]
    for column, (label, value, style) in zip(metrics, metric_values):
        with column:
            st.markdown(
                f'<div class="result-metric"><div class="result-metric-label">{label}</div>'
                f'<div class="result-metric-value {style}">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="results-panel"><h3>Distribución de las clasificaciones</h3>', unsafe_allow_html=True)
    # Se mantiene un orden fijo de clases, incluso cuando alguna tiene cero registros.
    distribution = analysis.records["clasificacion"].value_counts()
    ordered_classes = [
        "Normal", "Pitufeo", "Redondeo de cifras", "Fraude de nómina",
        "Registro de pasivos como ingresos", "Uso de cuentas inexistentes",
    ]
    maximum = max(distribution.max(), 1)
    bars = []
    for classification in ordered_classes:
        count = int(distribution.get(classification, 0))
        width = count / maximum * 100
        bar_class = "distribution-normal" if classification == "Normal" else "distribution-alert"
        bars.append(
            f'<div class="distribution-row"><div class="distribution-label">{classification}</div>'
            f'<div class="distribution-track"><div class="distribution-bar {bar_class}" style="width:{width:.2f}%"></div></div>'
            f'<div class="distribution-count">{count:,}</div></div>'
        )
    st.markdown("".join(bars) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="results-panel"><h3>Registros identificados</h3>', unsafe_allow_html=True)
    filters = st.columns([1, 1, 2])
    class_filter = filters[0].selectbox(
        "Clasificación", ["Todos"] + [value for value in ordered_classes if value in distribution],
        key="results_class_filter",
    )
    result_filter = filters[1].selectbox(
        "Resultado", ["Todos", "Normal", "Alerta"], key="results_result_filter"
    )
    search = filters[2].text_input(
        "Buscar registro", placeholder="Número de registro o comprobante", key="results_search"
    )

    # Los filtros afectan solo la vista y no destruyen el resultado completo descargable.
    filtered = analysis.records.copy()
    if class_filter != "Todos":
        filtered = filtered[filtered["clasificacion"] == class_filter]
    if result_filter != "Todos":
        filtered = filtered[filtered["resultado"] == result_filter]
    if search:
        term = search.lower()
        filtered = filtered[
            filtered.astype(str).apply(lambda row: row.str.lower().str.contains(term).any(), axis=1)
        ]

    # La paginación evita enviar miles de filas al navegador en un solo render.
    page_size = 5
    total_pages = max((len(filtered) + page_size - 1) // page_size, 1)
    page_key = "results_page"
    current_page = min(st.session_state.get(page_key, 1), total_pages)
    page = st.number_input(
        "Página", min_value=1, max_value=total_pages, value=current_page, step=1,
        label_visibility="collapsed", key="results_page_input",
    )
    st.session_state[page_key] = int(page)
    start = (int(page) - 1) * page_size
    page_records = filtered.iloc[start:start + page_size]
    st.caption(f"Mostrando {len(page_records):,} de {len(filtered):,} registros")

    display_data = {"Registro": page_records["registro"].tolist()}
    document_column = next(
        (column for column in ("tipo_comprobante", "Tipo de comprobante", "tipo", "Tipo") if column in page_records),
        None,
    )
    if document_column:
        display_data["Tipo de comprobante"] = page_records[document_column].astype(str).tolist()
    display_data.update({
        "Clasificación": page_records["clasificacion"].tolist(),
        "Probabilidad": [f"{value:.1%}" for value in page_records["probabilidad"]],
        "Resultado": page_records["resultado"].tolist(),
    })
    display = pd.DataFrame(display_data)
    st.dataframe(display, use_container_width=True, hide_index=True)
    if not filtered.empty:
        selected_record = st.selectbox(
            "Ver detalle del registro", page_records["registro"].tolist(), key="selected_result_record"
        )
        record = filtered[filtered["registro"] == selected_record].iloc[0]
        with st.expander(f"Detalle del registro {selected_record}"):
            st.write(f"**Clasificación:** {record['clasificacion']}")
            st.write(f"**Probabilidad:** {record['probabilidad']:.1%}")
            st.write(f"**Resultado:** {record['resultado']}")
            st.write("El modelo identificó patrones que deben ser revisados con el contexto contable original.")
    st.markdown("</div>", unsafe_allow_html=True)

    csv_data = analysis.records.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar resultados CSV", csv_data, "resultados_fraude.csv", "text/csv")


def render_config() -> None:
    st.title("Configuración")
    st.markdown('<p class="app-subtitle">Ajusta las preferencias del sistema.</p>', unsafe_allow_html=True)
    st.selectbox("Modelo predeterminado", ["LightGBM", "Random Forest", "XGBoost", "Decision Tree"])
    st.selectbox("Formato de descarga", ["CSV", "Excel", "Reporte"])
    appearance = st.radio(
        "Apariencia",
        ["Claro", "Oscuro", "Sistema"],
        index=["Claro", "Oscuro", "Sistema"].index(st.session_state.get("appearance", "Claro")),
        horizontal=True,
        key="appearance_selector",
    )
    st.session_state.appearance = appearance
    st.caption("Versión de la aplicación: 1.0 · Estado: Prototipo / En desarrollo")
