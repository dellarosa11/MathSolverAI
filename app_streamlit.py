from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st
from PIL import Image

from main import MathSolverAI


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def list_available_models() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.pth"))


@st.cache_resource(show_spinner=False)
def load_solver_app(model_path: str) -> MathSolverAI:
    return MathSolverAI(model_path)


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


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
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(126, 211, 180, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(61, 145, 255, 0.12), transparent 24%),
                linear-gradient(180deg, #f7faf8 0%, #eef4f1 100%);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(8px);
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

        model_names = [path.name for path in available_models]
        default_index = 0
        for index, name in enumerate(model_names):
            if "v5_best" in name:
                default_index = index
                break

        selected_model_name = st.selectbox("Modelo", model_names, index=default_index)
        selected_model_path = MODELS_DIR / selected_model_name

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
            <p style="margin:0.5rem 0 0 0; color:#334155;">
                Envie uma foto de conta, expressao ou equacao. O app reconhece os simbolos,
                tenta corrigir ambiguidades e mostra o resultado em formato de conversa.
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
        st.image(preview, caption=f"Prévia: {uploaded_file.name}", use_container_width=True)


if __name__ == "__main__":
    main()
