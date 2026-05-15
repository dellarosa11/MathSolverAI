from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from main import MathSolverAI


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_LABELS = {
    "modelo_principal_fotos_reais_v5_best.pth": "Modelo principal - fotos reais (V5)",
    "modelo_alternativo_bhmsds_v4_best.pth": "Modelo alternativo - BHMSDS (V4)",
}


def list_available_models() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.pth"))


def build_model_options(models: list[Path]) -> tuple[list[str], dict[str, Path]]:
    labels: list[str] = []
    mapping: dict[str, Path] = {}

    def sort_key(model_path: Path) -> tuple[int, str]:
        name = model_path.name
        if name == "modelo_principal_fotos_reais_v5_best.pth":
            return (0, name)
        if name == "modelo_alternativo_bhmsds_v4_best.pth":
            return (1, name)
        return (2, name)

    for model_path in sorted(models, key=sort_key):
        friendly_label = MODEL_LABELS.get(model_path.name, model_path.name)
        label = f"{friendly_label} [{model_path.name}]"
        labels.append(label)
        mapping[label] = model_path

    return labels, mapping


@st.cache_resource(show_spinner=False)
def load_solver_app(model_path: str) -> MathSolverAI:
    return MathSolverAI(model_path)


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def image_to_png_bytes(image: np.ndarray) -> bytes:
    if image.ndim == 2:
        success, encoded = cv2.imencode(".png", image)
    else:
        success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Falha ao converter imagem para PNG em memoria.")
    return encoded.tobytes()


def build_annotated_image(original_image: np.ndarray, symbols) -> np.ndarray:
    if original_image.ndim == 2:
        canvas = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    for index, symbol in enumerate(symbols, start=1):
        x, y, w, h = symbol.box
        label = f"{index}:{symbol.label} {symbol.confidence * 100:.1f}%"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (72, 201, 176), 2)
        cv2.putText(
            canvas,
            label,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 191, 87),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_processed_detection_image(binary_image: np.ndarray, symbols) -> np.ndarray:
    canvas = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

    for index, symbol in enumerate(symbols, start=1):
        x, y, w, h = symbol.box
        label = f"S{index}:{symbol.label}"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 120, 80), 2)
        cv2.putText(
            canvas,
            label,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_prepared_symbol_gallery(binary_image: np.ndarray, recognition, processor) -> list[dict[str, object]]:
    gallery: list[dict[str, object]] = []
    for index, symbol in enumerate(recognition.symbols, start=1):
        x, y, w, h = symbol.box
        raw_crop = binary_image[y:y + h, x:x + w]
        nn_crop = processor.prepare_for_nn(raw_crop)
        gallery.append(
            {
                "index": index,
                "label": symbol.label,
                "confidence": float(symbol.confidence),
                "raw_crop_bytes": image_to_png_bytes(raw_crop),
                "nn_crop_bytes": image_to_png_bytes(nn_crop),
            }
        )
    return gallery


def build_symbol_diagnostics(recognition) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for index, symbol in enumerate(recognition.symbols, start=1):
        diagnostics.append(
            {
                "index": index,
                "label": symbol.label,
                "confidence": float(symbol.confidence),
                "box": symbol.box,
                "top_predictions": symbol.top_predictions,
            }
        )
    return diagnostics


def analyze_image(
    *,
    image_path: Path,
    model_path: Path,
    top_k: int,
    use_correction: bool,
    beam_width: int,
    max_candidates: int,
) -> dict[str, object]:
    app = load_solver_app(str(model_path))
    processing_debug = app.processor.get_processing_debug(image_path)
    original_image = processing_debug["original"]
    binary_image = processing_debug["final_binary"]

    with redirect_stdout(io.StringIO()):
        recognition = app.recognize_expression(image_path, top_k=top_k)
        correction = app.improve_expression(
            recognition,
            use_correction=use_correction,
            beam_width=beam_width,
            alternatives_per_symbol=top_k,
            max_candidates=max_candidates,
        )

    expression_to_use = correction.corrected_expression
    normalized_expression = None
    solution_text = None
    solution_kind = None
    error_message = None
    solution_details: dict[str, object] | None = None

    try:
        normalized_expression = app.solver.normalize_expression(expression_to_use)
        if hasattr(app.solver, "describe_solution"):
            solution_details = app.solver.describe_solution(normalized_expression)
            solution_kind = str(solution_details.get("kind", "expression"))
            result = solution_details.get("result")
            if solution_kind == "numeric_equation":
                left_side = solution_details.get("left_side")
                right_side = solution_details.get("right_side")
                left_value = solution_details.get("left_value")
                right_value = solution_details.get("right_value")
                solution_text = (
                    f"Igualdade: **{result}**\n\n"
                    f"- Lado esquerdo: `{left_side}` = `{left_value}`\n"
                    f"- Lado direito: `{right_side}` = `{right_value}`"
                )
            elif solution_kind == "symbolic_equation":
                solution_text = f"Solucao: **{result}**"
            else:
                solution_text = f"Resultado: **{result}**"
        else:
            result = app.solver.solve(normalized_expression)
            solution_kind = "expression"
            solution_text = f"Resultado: **{result}**"
    except Exception as exc:
        error_message = str(exc)

    low_confidence = [
        item
        for item in build_symbol_diagnostics(recognition)
        if float(item["confidence"]) < 0.7
    ]
    annotated_image = build_annotated_image(original_image, recognition.symbols)
    processed_detection_image = build_processed_detection_image(binary_image, recognition.symbols)
    symbol_gallery = build_prepared_symbol_gallery(binary_image, recognition, app.processor)
    processing_stage_bytes = {
        name: image_to_png_bytes(stage)
        for name, stage in processing_debug.items()
    }

    return {
        "image_path": str(image_path),
        "recognized_expression": recognition.expression,
        "corrected_expression": correction.corrected_expression,
        "correction_changed": correction.changed,
        "normalized_expression": normalized_expression,
        "solution_text": solution_text,
        "solution_kind": solution_kind,
        "solution_details": solution_details,
        "error_message": error_message,
        "symbols": build_symbol_diagnostics(recognition),
        "low_confidence_symbols": low_confidence,
        "correction_candidates": correction.candidates,
        "original_image_bytes": image_to_png_bytes(original_image),
        "binary_image_bytes": image_to_png_bytes(binary_image),
        "processing_stages": processing_stage_bytes,
        "processed_detection_bytes": image_to_png_bytes(processed_detection_image),
        "annotated_image_bytes": image_to_png_bytes(annotated_image),
        "symbol_gallery": symbol_gallery,
    }


def render_symbol_list(symbols: list[dict[str, object]]) -> None:
    for symbol in symbols:
        top_predictions = ", ".join(
            f"{candidate['label']} ({candidate['confidence'] * 100:.1f}%)"
            for candidate in symbol["top_predictions"]
        )
        st.markdown(
            (
                f"**S{symbol['index']}**  \n"
                f"rotulo: `{symbol['label']}`  \n"
                f"confianca: `{symbol['confidence'] * 100:.1f}%`  \n"
                f"caixa: `{symbol['box']}`  \n"
                f"top-k: {top_predictions}"
            )
        )


def render_correction_candidates(candidates) -> None:
    for index, candidate in enumerate(candidates, start=1):
        st.markdown(
            (
                f"**C{index}** `{candidate.expression}`  \n"
                f"score: `{candidate.score:.3f}`  \n"
                f"valida: `{'sim' if candidate.valid else 'nao'}`  \n"
                f"resolvivel: `{'sim' if candidate.solvable else 'nao'}`"
            )
        )


def render_symbol_gallery(symbol_gallery: list[dict[str, object]]) -> None:
    for item in symbol_gallery:
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                item["raw_crop_bytes"],
                caption=f"S{item['index']} bruto - {item['label']}",
                use_container_width=True,
            )
        with col2:
            st.image(
                item["nn_crop_bytes"],
                caption=f"S{item['index']} pronto para a rede",
                use_container_width=True,
            )
        st.caption(f"Confianca: {item['confidence'] * 100:.1f}%")


def render_processing_stages(processing_stages: dict[str, bytes]) -> None:
    stage_labels = {
        "original": "Original",
        "denoised": "Reducao de ruido",
        "contrast": "Contraste reforcado",
        "line_mask": "Linhas detectadas",
        "line_removed": "Caderno sem linhas",
        "normalized": "Correcao de iluminacao",
        "blurred": "Suavizacao",
        "threshold": "Threshold inicial",
        "opened": "Remocao de ruido fino",
        "closed": "Fechamento de falhas",
        "final_binary": "Resultado final tratado",
    }

    stage_items = [
        (name, processing_stages[name])
        for name in stage_labels
        if name in processing_stages
    ]
    for start_index in range(0, len(stage_items), 2):
        row_items = stage_items[start_index:start_index + 2]
        columns = st.columns(len(row_items))
        for column, (name, image_bytes) in zip(columns, row_items):
            with column:
                st.image(
                    image_bytes,
                    caption=stage_labels.get(name, name),
                    use_container_width=True,
                )


def render_pipeline_visuals(analysis: dict[str, object]) -> None:
    tabs = st.tabs(["Entrada", "Tratamento", "Onde a IA esta vendo", "Resultado final", "Recortes"])
    with tabs[0]:
        st.image(
            analysis["original_image_bytes"],
            caption="Imagem original enviada",
            use_container_width=True,
        )
    with tabs[1]:
        st.caption("Etapas do tratamento aplicadas na foto antes da leitura dos simbolos.")
        render_processing_stages(analysis["processing_stages"])
    with tabs[2]:
        st.image(
            analysis["processed_detection_bytes"],
            caption="Imagem tratada com as caixas dos simbolos que a IA realmente recortou",
            use_container_width=True,
        )
        st.caption("Use esta tela para validar se a IA esta enxergando os numeros e operadores nos lugares certos.")
    with tabs[3]:
        st.image(
            analysis["annotated_image_bytes"],
            caption="Resultado visual com caixas, rotulos e confiancas",
            use_container_width=True,
        )
    with tabs[4]:
        render_symbol_gallery(analysis["symbol_gallery"])


def push_message(role: str, content: str, **extra: object) -> None:
    st.session_state.chat_history.append({"role": role, "content": content, **extra})


def reset_chat() -> None:
    st.session_state.chat_history = []


def ensure_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Envie uma imagem com uma conta ou equacao matematica. "
                    "Eu reconheco os simbolos, aplico o corretor e mostro o resultado."
                ),
            }
        ]


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            image_bytes = message.get("image_bytes")
            if image_bytes is not None:
                st.image(image_bytes, caption=message.get("image_name"), use_container_width=True)

            analysis = message.get("analysis")
            if not analysis:
                continue

            if analysis["recognized_expression"]:
                st.markdown(f"**Expressao reconhecida:** `{analysis['recognized_expression']}`")
            if analysis["correction_changed"]:
                st.markdown(f"**Corretor sugeriu:** `{analysis['corrected_expression']}`")
            elif analysis["corrected_expression"]:
                st.markdown(f"**Expressao final:** `{analysis['corrected_expression']}`")

            if analysis["solution_text"]:
                st.markdown(analysis["solution_text"])
            if analysis["error_message"]:
                st.error(f"Falha ao resolver: {analysis['error_message']}")

            with st.expander("Fluxo visual da imagem", expanded=True):
                render_pipeline_visuals(analysis)

            with st.expander("Diagnostico por simbolo", expanded=False):
                render_symbol_list(analysis["symbols"])

            if analysis["low_confidence_symbols"]:
                with st.expander("Simbolos com baixa confianca", expanded=False):
                    render_symbol_list(analysis["low_confidence_symbols"])

            with st.expander("Candidatos do corretor", expanded=False):
                render_correction_candidates(analysis["correction_candidates"])


def main() -> None:
    st.set_page_config(
        page_title="MathSolverAI Chat",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_state()

    st.markdown(
        """
        <style>
        :root {
            --app-glass-light: rgba(255, 255, 255, 0.80);
            --app-glass-dark: rgba(15, 23, 42, 0.84);
            --app-border-light: rgba(15, 23, 42, 0.10);
            --app-border-dark: rgba(148, 163, 184, 0.18);
            --panel-light: rgba(255, 255, 255, 0.92);
            --panel-dark: rgba(17, 24, 39, 0.92);
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(126, 211, 180, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(61, 145, 255, 0.12), transparent 24%),
                linear-gradient(180deg, var(--background-color, #f7faf8) 0%, var(--secondary-background-color, #eef4f1) 100%);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: var(--app-glass-light);
            color: inherit;
            border: 1px solid var(--app-border-light);
            border-radius: 22px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(8px);
        }
        .hero-card h1,
        .hero-card p,
        .hero-card strong,
        .hero-card span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] * {
            color: inherit !important;
        }
        [data-testid="stChatMessage"] {
            border-radius: 20px;
            background: var(--panel-light);
            border: 1px solid rgba(148, 163, 184, 0.15);
        }
        [data-testid="stExpander"] {
            border-radius: 18px;
            overflow: hidden;
            background: var(--panel-light);
        }
        [data-testid="stSidebar"] {
            background: rgba(248, 250, 252, 0.92);
        }
        html[data-theme="dark"] .hero-card,
        body[data-theme="dark"] .hero-card {
            background: var(--app-glass-dark);
            border-color: var(--app-border-dark);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
        }
        html[data-theme="dark"] [data-testid="stChatMessage"],
        body[data-theme="dark"] [data-testid="stChatMessage"],
        html[data-theme="dark"] [data-testid="stExpander"],
        body[data-theme="dark"] [data-testid="stExpander"] {
            background: var(--panel-dark);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        html[data-theme="dark"] [data-testid="stSidebar"],
        body[data-theme="dark"] [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.96);
        }
        html[data-theme="dark"] .stApp,
        html[data-theme="dark"] [data-testid="stAppViewContainer"],
        body[data-theme="dark"] .stApp,
        body[data-theme="dark"] [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.13), transparent 26%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.15), transparent 26%),
                linear-gradient(180deg, #0b1220 0%, #111827 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    available_models = list_available_models()
    if not available_models:
        st.error("Nenhum modelo .pth foi encontrado na pasta models/.")
        st.stop()

    with st.sidebar:
        st.title("MathSolverAI")
        st.caption("Interface em estilo chat para testar a IA com imagens reais.")

        model_labels, model_mapping = build_model_options(available_models)
        default_index = 0
        for index, label in enumerate(model_labels):
            if "modelo_principal_fotos_reais_v5_best.pth" in label:
                default_index = index
                break

        selected_model_label = st.selectbox("Modelo", model_labels, index=default_index)
        selected_model_path = model_mapping[selected_model_label]
        selected_model_name = selected_model_path.name

        if selected_model_name in MODEL_LABELS:
            st.caption(f"Selecionado: {MODEL_LABELS[selected_model_name]}")

        top_k = st.slider("Top-k de diagnostico", min_value=3, max_value=10, value=10)
        beam_width = st.slider("Beam width do corretor", min_value=8, max_value=40, value=24, step=4)
        max_candidates = st.slider("Maximo de candidatos", min_value=3, max_value=12, value=8)
        use_correction = st.toggle("Usar corretor de expressao", value=True)

        st.divider()
        uploaded_file = st.file_uploader(
            "Envie uma imagem",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )
        if st.button("Limpar conversa", use_container_width=True):
            reset_chat()
            st.rerun()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0; font-size:2rem;">MathSolverAI Chat</h1>
            <p style="margin:0.5rem 0 0 0;">
                Envie uma foto de conta, expressao ou equacao. O app reconhece os simbolos,
                tenta corrigir ambiguidades e mostra o resultado em formato de conversa.
                Agora a interface tambem exibe a imagem original, cada etapa do tratamento e o resultado visual final.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    render_chat_history()

    prompt = st.chat_input("Escreva algo como 'resolva esta conta' e envie uma imagem na barra lateral.")
    if prompt:
        push_message("user", prompt)

    if uploaded_file is not None and st.button("Analisar imagem", type="primary", use_container_width=False):
        image_bytes = uploaded_file.getvalue()
        user_prompt = prompt or f"Resolva esta imagem: {uploaded_file.name}"
        push_message(
            "user",
            user_prompt,
            image_bytes=image_bytes,
            image_name=uploaded_file.name,
        )

        temp_image_path = save_uploaded_file(uploaded_file)
        try:
            analysis = analyze_image(
                image_path=temp_image_path,
                model_path=selected_model_path,
                top_k=top_k,
                use_correction=use_correction,
                beam_width=beam_width,
                max_candidates=max_candidates,
            )
        finally:
            temp_image_path.unlink(missing_ok=True)

        summary_lines = [
            f"Modelo usado: `{selected_model_name}`",
            f"Expressao reconhecida: `{analysis['recognized_expression'] or '(vazia)'}`",
        ]
        if analysis["correction_changed"]:
            summary_lines.append(f"Expressao corrigida: `{analysis['corrected_expression']}`")
        if analysis["solution_text"]:
            summary_lines.append(analysis["solution_text"])
        if analysis["error_message"]:
            summary_lines.append(f"Falha ao resolver: `{analysis['error_message']}`")

        push_message(
            "assistant",
            "\n\n".join(summary_lines),
            analysis=analysis,
        )
        st.rerun()

    if uploaded_file is not None:
        st.write("")
        preview = Image.open(uploaded_file)
        st.image(preview, caption=f"Previa: {uploaded_file.name}", use_container_width=True)


if __name__ == "__main__":
    main()
