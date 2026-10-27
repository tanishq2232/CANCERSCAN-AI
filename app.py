
from pathlib import Path
import io
import json
import sqlite3
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "outputs" / "prediction_history.db"

MODEL_CONFIG = {
    "Lung": {
        "path": ROOT / "saved_models" / "lung" /
                "resnet18_224_baseline_best.pth",
        "architecture": "resnet18",
        "image_size": 224,
        "labels": ["Cancer", "NORMAL"],
        "description": "Chest X-ray classification",
    },
    "Breast": {
        # Update this path after verifying the actual
        # breast model architecture and checkpoint format.
        "path": ROOT / "saved_models" / "breast" /
                "breast_resnet18_best.pth",
        "architecture": "resnet18",
        "image_size": 224,
        "labels": ["Benign", "Malignant"],
        "description": "Breast image classification",
    },
}

MAX_UPLOAD_MB = 15
MEAN_DEFAULT = [0.485, 0.456, 0.406]
STD_DEFAULT = [0.229, 0.224, 0.225]

(ROOT / "outputs").mkdir(parents=True, exist_ok=True)


# =========================================================
# PAGE + THEME
# =========================================================

st.set_page_config(
    page_title="CancerScan-AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

dark = st.session_state.theme == "Dark"

if dark:
    bg = "#111827"
    panel = "#1f2937"
    ink = "#f3f4f6"
    muted = "#aab5c5"
    border = "#374151"
    accent = "#a5b4fc"
    hero = "#202b40"
else:
    bg = "#f5f6fb"
    panel = "#ffffff"
    ink = "#20283a"
    muted = "#667085"
    border = "#e5e7ef"
    accent = "#635bdb"
    hero = "#efefff"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .stApp {{
        background: {bg};
        color: {ink};
    }}

    [data-testid="stSidebar"] {{
        background: {panel};
        border-right: 1px solid {border};
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }}

    .hero {{
        background: {hero};
        border: 1px solid {border};
        border-radius: 22px;
        padding: 27px;
        margin-bottom: 22px;
    }}

    .eyebrow {{
        color: {accent};
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 800;
    }}

    .hero h1 {{
        font-family: 'Manrope', sans-serif;
        font-size: clamp(28px, 4vw, 42px);
        color: {ink};
        margin: 8px 0;
    }}

    .hero p, .muted {{
        color: {muted};
    }}

    .card {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 17px;
        padding: 19px;
        margin-bottom: 12px;
    }}

    .metric {{
        color: {muted};
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .metric-value {{
        color: {ink};
        font: 800 27px 'Manrope', sans-serif;
        margin-top: 7px;
    }}

    .stButton > button {{
        border-radius: 11px;
        min-height: 43px;
        font-weight: 700;
    }}

    div[data-testid="stFileUploader"] {{
        border: 1px dashed {border};
        border-radius: 15px;
        padding: 12px;
    }}

    h1, h2, h3, p, label {{
        overflow-wrap: anywhere;
    }}

    .footer {{
        color: {muted};
        text-align: center;
        font-size: 12px;
        padding-top: 25px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SQLITE HISTORY
# =========================================================

def connect_db():
    return sqlite3.connect(DB_PATH, timeout=10)


def initialize_db():
    with connect_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                module TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                inference_seconds REAL NOT NULL,
                model_path TEXT NOT NULL
            )
        """)


def save_prediction(
    module, filename, predicted_class,
    scores, inference_seconds, model_path
):
    with connect_db() as con:
        con.execute("""
            INSERT INTO predictions (
                timestamp, module, filename, predicted_class,
                scores_json, inference_seconds, model_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            module,
            filename,
            predicted_class,
            json.dumps(scores),
            round(inference_seconds, 4),
            str(model_path),
        ))


def load_history():
    with connect_db() as con:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            con,
        )


initialize_db()


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path_string, architecture):
    path = Path(model_path_string)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required checkpoint is missing: {path.name}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Only load trusted, locally generated checkpoints.
    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint could not be safely loaded. "
            "Verify its format and recreate it as a state_dict "
            "or a compatible weights-only checkpoint."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected checkpoint format.")

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        labels = checkpoint.get("class_names")
        image_size = int(checkpoint.get("image_size", 224))
        mean = checkpoint.get("mean", MEAN_DEFAULT)
        std = checkpoint.get("std", STD_DEFAULT)
    else:
        state = checkpoint
        labels = None
        image_size = 224
        mean, std = MEAN_DEFAULT, STD_DEFAULT

    if not labels:
        raise ValueError(
            "Checkpoint does not include class_names. "
            "Configure verified class labels for this model."
        )

    if architecture != "resnet18":
        raise ValueError("Unsupported configured architecture.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    preprocessing = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, device, list(labels), image_size, preprocessing


# =========================================================
# INFERENCE
# =========================================================

def run_inference(image, model, device, labels, preprocessing):
    tensor = preprocessing(image).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(tensor)

        if logits.ndim != 2 or logits.shape[1] != len(labels):
            raise ValueError("Model output does not match class labels.")

        scores = torch.softmax(logits, dim=1)[0].cpu().tolist()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    ranked = sorted(
        zip(labels, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked, elapsed


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("## 🩺 CancerScan-AI")
    st.caption("AI-Powered Medical Image Analysis")
    st.divider()

    st.markdown("### Appearance")
    theme_choice = st.radio(
        "Choose theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    page = st.radio(
        "Navigation",
        [
            "Home Dashboard",
            "Breast Detection",
            "Lung Detection",
            "Prediction History",
            "Model Information",
            "About",
        ],
    )

    st.divider()
    st.caption("Local research prototype")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Medical AI Research Workspace</div>
        <h1>CancerScan-AI</h1>
        <p>
            A research-oriented interface for breast and lung
            image classification using locally stored models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Home Dashboard":
    history = load_history()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="card"><div class="metric">'
            f'Detection Modules</div><div class="metric-value">2</div>'
            f'<p class="muted">Breast and lung</p></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div class="card"><div class="metric">'
            f'Analyses Saved</div><div class="metric-value">'
            f'{len(history)}</div><p class="muted">'
            f'Persistent local history</p></div>',
            unsafe_allow_html=True,
        )

    with c3:
        latest = (
            f"{history.iloc[0]['inference_seconds']:.3f} s"
            if not history.empty else "—"
        )
        st.markdown(
            f'<div class="card"><div class="metric">'
            f'Latest Inference</div><div class="metric-value">'
            f'{latest}</div><p class="muted">'
            f'Measured model inference</p></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Choose a detection module")
    a, b = st.columns(2)

    with a:
        st.markdown(
            '<div class="card"><h3>🫁 Lung Detection</h3>'
            '<p class="muted">Chest X-ray classification using '
            'the configured ResNet18 checkpoint.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open Lung Module", use_container_width=True):
            st.session_state["requested_page"] = "Lung Detection"
            st.rerun()

    with b:
        st.markdown(
            '<div class="card"><h3>🎗️ Breast Detection</h3>'
            '<p class="muted">Mammogram analysis, when a compatible '
            'breast model is configured.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open Breast Module", use_container_width=True):
            st.session_state["requested_page"] = "Breast Detection"
            st.rerun()

    requested = st.session_state.pop("requested_page", None)
    if requested:
        st.info(f"Select **{requested}** from the sidebar to continue.")

    st.subheader("Recent analyses")
    if history.empty:
        st.info("No predictions saved yet.")
    else:
        st.dataframe(
            history[[
                "timestamp", "module", "filename",
                "predicted_class", "inference_seconds"
            ]].head(5),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# BREAST AND LUNG MODULES
# =========================================================

elif page in ("Breast Detection", "Lung Detection"):
    module = "Breast" if page == "Breast Detection" else "Lung"
    cfg = MODEL_CONFIG[module]

    st.subheader(f"{'🎗️' if module == 'Breast' else '🫁'} {module} Image Analysis")
    st.caption(cfg["description"])

    if not cfg["path"].is_file():
        st.error(
            f"The configured {module.lower()} model is not available. "
            f"Expected filename: `{cfg['path'].name}`"
        )

        if module == "Breast":
            st.warning(
                "Update MODEL_CONFIG with the real breast checkpoint, "
                "architecture, input size, and class mapping. The "
                "breast model must be verified before enabling predictions."
            )
        else:
            st.info(
                "Train the ResNet18 baseline first, or place its "
                "compatible checkpoint at the configured path."
            )

        st.stop()

    try:
        model, device, labels, image_size, preprocessing = load_model(
            str(cfg["path"]), cfg["architecture"]
        )
    except Exception as exc:
        st.error(
            "The model could not be loaded. Check the checkpoint "
            "format, class names, architecture, and configuration."
        )
        st.caption(str(exc))
        st.stop()

    st.caption(
        f"Device: {device} · Input: {image_size} × {image_size} · "
        f"Classes: {', '.join(labels)}"
    )

    upload = st.file_uploader(
        f"Upload {module.lower()} image",
        type=["png", "jpg", "jpeg"],
        key=f"upload_{module}",
        help="Maximum file size: 15 MB.",
    )

    if upload is not None:
        if len(upload.getvalue()) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error("Image exceeds the 15 MB upload limit.")
            st.stop()

        try:
            image = ImageOps.exif_transpose(
                Image.open(io.BytesIO(upload.getvalue()))
            ).convert("RGB")
        except Exception:
            st.error("This file could not be decoded as an image.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption=f"Uploaded image · {image.width} × {image.height}",
                use_container_width=True,
            )

        with col2:
            preview = image.convert("L").convert("RGB").resize(
                (image_size, image_size)
            )
            st.image(
                preview,
                caption=f"Model preview · {image_size} × {image_size}",
                use_container_width=True,
            )

        if st.button(
            f"Analyze {module} Image",
            type="primary",
            use_container_width=True,
            key=f"analyze_{module}",
        ):
            with st.spinner("Running model inference..."):
                try:
                    results, elapsed = run_inference(
                        image, model, device, labels, preprocessing
                    )
                except Exception:
                    st.error(
                        "Prediction failed. Verify the model's "
                        "input requirements and output format."
                    )
                    st.stop()

            predicted_class, top_score = results[0]
            st.session_state["last_result"] = {
                "module": module,
                "filename": Path(upload.name).name,
                "predicted_class": predicted_class,
                "scores": dict(results),
                "elapsed": elapsed,
            }

            try:
                save_prediction(
                    module=module,
                    filename=Path(upload.name).name,
                    predicted_class=predicted_class,
                    scores=dict(results),
                    inference_seconds=elapsed,
                    model_path=cfg["path"],
                )
            except sqlite3.Error:
                st.error("Prediction completed, but history could not be saved.")
                st.stop()

            st.success("Prediction completed and history saved.")

        last = st.session_state.get("last_result")

        if last and last["module"] == module:
            st.divider()
            st.subheader("Analysis Result")

            st.markdown(
                f'<div class="card"><div class="metric">'
                f'Model output · {module}</div>'
                f'<h2>{last["predicted_class"]}</h2>'
                f'<p class="muted">Highest model score: '
                f'{max(last["scores"].values()):.2%}</p>'
                f'<p class="muted">Inference time: '
                f'{last["elapsed"]:.3f} seconds</p></div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Class scores")
            for label, score in last["scores"].items():
                st.write(f"**{label}** — {score:.2%}")
                st.progress(float(score))


# =========================================================
# HISTORY
# =========================================================

elif page == "Prediction History":
    st.subheader("📋 Prediction History")
    st.caption(f"Stored locally in `{DB_PATH.name}`. Images are not saved.")

    history = load_history()

    if history.empty:
        st.info("No prediction history available yet.")
    else:
        display = history.drop(columns=["scores_json", "model_path"])
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.download_button(
            "Download History as CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="cancerscan_prediction_history.csv",
            mime="text/csv",
        )

        with st.expander("View saved class scores"):
            for _, row in history.iterrows():
                st.write(
                    f"**{row['timestamp']} · {row['module']} · "
                    f"{row['filename']}**"
                )
                st.json(json.loads(row["scores_json"]))

        if st.button("Delete All History", type="secondary"):
            with connect_db() as con:
                con.execute("DELETE FROM predictions")
            st.success("Local prediction history deleted.")
            st.rerun()


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "Model Information":
    st.subheader("Model Information")

    for module, cfg in MODEL_CONFIG.items():
        exists = cfg["path"].is_file()

        with st.expander(f"{module} · {cfg['architecture']}"):
            st.write("**Architecture:**", cfg["architecture"])
            st.write("**Input size:**", f"{cfg['image_size']} × {cfg['image_size']}")
            st.write("**Configured classes:**", ", ".join(cfg["labels"]))
            st.write("**Checkpoint available:**", "Yes" if exists else "No")
            st.write("**Model file:**", cfg["path"].name)

            if not exists:
                st.warning("No compatible checkpoint found at the configured path.")

    st.info(
        "No accuracy values are shown here because verified evaluation "
        "metrics have not been configured in this application."
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":
    st.subheader("About CancerScan-AI")

    st.markdown("""
    CancerScan-AI is an educational and research-oriented application
    for image classification experiments.

    **Technology stack**
    - Python and Streamlit
    - PyTorch and Torchvision
    - Pillow for image handling
    - SQLite and Pandas for prediction history

    **Privacy**
    - Uploaded image contents are not stored in the history database.
    - History records contain filenames, predicted classes, scores,
      timestamps, model paths, and inference durations.
    - The SQLite database is stored locally in the project's `outputs`
      directory.
    """)

    st.warning(
        "Research and educational use only. This application is not a "
        "medical diagnostic device. AI-generated results must not be used "
        "independently to make medical decisions. Consult a qualified "
        "healthcare professional."
    )


st.markdown(
    '<div class="footer">CancerScan-AI · Research prototype · '
    'Not for clinical diagnosis</div>',
    unsafe_allow_html=True,
)
