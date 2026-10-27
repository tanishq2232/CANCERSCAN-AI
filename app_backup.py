
from pathlib import Path
from io import BytesIO
import hashlib
import html
import time

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps, UnidentifiedImageError

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

LUNG_MODEL_PATH = (
    ROOT / "saved_models" / "lung_text_removed"
    / "resnet18_text_removed_best.pth"
)

# Placeholder: set this to your actual compatible breast model.
BREAST_MODEL_PATH = ROOT / "saved_models" / "breast_resnet18_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MAX_IMAGE_PIXELS = 25_000_000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CLASSES = ["Cancer", "NORMAL"]

st.set_page_config(
    page_title="CancerScan AI",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. SESSION STATE
# ============================================================

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if "page" not in st.session_state:
    st.session_state.page = "Home"

for scan_name in ("Lung Scan", "Breast Scan"):
    prefix = scan_name.lower().replace(" ", "_")
    st.session_state.setdefault(f"{prefix}_result", None)
    st.session_state.setdefault(f"{prefix}_hash", None)

IS_LIGHT = st.session_state.theme_mode == "Light"

# ============================================================
# 3. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">CancerScan <span>AI</span></div>
        <div class="muted">Medical imaging research workspace</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.radio(
        "APPEARANCE",
        ["Dark", "Light"],
        horizontal=True,
        key="theme_mode",
    )
    st.divider()
    st.radio(
        "WORKSPACE",
        ["Home", "Lung Scan", "Breast Scan"],
        key="page",
    )
    st.divider()
    st.markdown("### Workspace information")
    st.markdown(
        f"""
        <div class="muted">
        Compute device<br><b>{DEVICE.type.upper()}</b><br><br>
        Model input size<br><b>{IMAGE_SIZE} × {IMAGE_SIZE}</b><br><br>
        Selected appearance<br><b>{st.session_state.theme_mode}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div class="muted">
        Research prototype only.<br>
        Not a clinical diagnostic system.
        </div>
        """,
        unsafe_allow_html=True,
    )

IS_LIGHT = st.session_state.theme_mode == "Light"

# ============================================================
# 4. THEME CSS
# ============================================================

DARK_CSS = """
:root {
 --bg:#0B0B10;--surface:#14141E;--surface2:#1B1928;
 --purple:#8657F5;--purple-light:#B49AFF;--yellow:#F5E642;
 --text:#F6F4FC;--muted:#B8B3C9;--border:#39324D;
}
.stApp{background:var(--bg)!important;color:var(--text)!important}
[data-testid="stHeader"]{background:rgba(11,11,16,.96)!important}
[data-testid="stSidebar"]{background:#111019!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--text)}
.hero{background:radial-gradient(circle at 88% 10%,#39246D 0%,transparent 42%),linear-gradient(120deg,#171421,#201638 58%,#171421);border:1px solid #493766}
.hero-title,.feature-title{color:#fff!important}
.hero-copy{color:#D2CCE2!important}
.feature-card{background:linear-gradient(145deg,#171521,#211A33);border:1px solid #403451}
.panel,[data-testid="stMetric"]{background:var(--surface)!important;border-color:var(--border)!important}
.section-title,.score-row,.result-note{color:var(--text)!important}
.section-copy,.feature-copy,.muted,.empty-copy{color:var(--muted)!important}
.tag{background:#29203D;border:1px solid #57427F;color:#E1D6FF}
.eyebrow,.eyebrow-small{color:var(--yellow)}
.result-cancer{background:#2B1928;border:1px solid #78405D}
.result-normal{background:#17251E;border:1px solid #315B43}
.result-label{color:var(--muted)}
.result-name,[data-testid="stMetricValue"]{color:var(--yellow)!important}
.empty-state{background:#13121B;border:1px dashed #493A63}
.notice{background:#211D31;border:1px solid #493A63;color:#DDD4F0}
.footer{color:#B8B3C9;border-color:var(--border)}
[data-testid="stFileUploader"]{background:#111019;border:1px dashed #57427F}
[data-testid="stFileUploader"] *{color:var(--text)}
input,textarea{color:var(--text)!important;background:#171521!important}
[data-testid="stSelectbox"] *{color:var(--text)}
"""

LIGHT_CSS = """
:root {
 --bg:#F5F3FC;--surface:#FFFFFF;--surface2:#EEE9FA;
 --purple:#5B21B6;--purple-light:#6D28D9;--yellow:#F5E642;
 --text:#20152F;--muted:#514765;--border:#D8D0E8;
}
.stApp{background:#F5F3FC!important;color:#20152F!important}
[data-testid="stHeader"]{background:rgba(245,243,252,.96)!important}
[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid #D8D0E8!important}
[data-testid="stSidebar"] *{color:#20152F}
.hero{background:linear-gradient(125deg,#fff,#EDE5FF);border:1px solid #D5C4FA}
.hero-title,.hero-copy,.feature-title{color:#20152F!important}
.hero-title span,.eyebrow,.eyebrow-small{color:#5B21B6!important}
.feature-card,.panel,[data-testid="stMetric"]{background:#fff!important;border-color:#D8D0E8!important}
.section-title,.score-row,.result-note{color:#20152F!important}
.section-copy,.feature-copy,.muted,.empty-copy{color:#514765!important}
.tag{background:#EEE7FF;border:1px solid #C4B5FD;color:#4C1D95}
.result-cancer{background:#FFF0F2;border:1px solid #E8AAB6}
.result-normal{background:#EAF8EF;border:1px solid #9BC9A8}
.result-label{color:#514765}
.result-name,[data-testid="stMetricValue"]{color:#5B21B6!important}
.empty-state{background:#fff;border:1px dashed #C4B5FD}
.notice{background:#EEE7FF;border:1px solid #C4B5FD;color:#35204F}
.footer{color:#514765;border-color:#D8D0E8}
[data-testid="stFileUploader"]{background:#fff;border:1px dashed #A78BFA}
[data-testid="stFileUploader"] *,input,textarea{color:#20152F!important}
input,textarea{background:#fff!important}
[data-testid="stSelectbox"] *{color:#20152F}
"""

theme_css = LIGHT_CSS if IS_LIGHT else DARK_CSS

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    html,body,[class*="css"]{{font-family:'DM Sans',sans-serif}}
    #MainMenu,footer{{visibility:hidden}}
    .block-container{{width:100%;max-width:1250px;padding-top:1.5rem;padding-bottom:2rem}}
    .brand{{font-family:'Manrope',sans-serif;font-weight:800;font-size:24px;letter-spacing:-.8px}}
    .brand span{{color:#F5E642}}
    .muted{{font-size:12px;line-height:1.8}}
    .hero{{border-radius:22px;padding:clamp(22px,4vw,38px);margin:4px 0 25px}}
    .eyebrow{{font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}}
    .hero-title{{font-family:'Manrope',sans-serif;font-size:clamp(29px,4vw,46px);font-weight:800;line-height:1.15;letter-spacing:-1.2px;margin-bottom:14px;overflow-wrap:anywhere}}
    .hero-title span{{color:#F5E642}}
    .hero-copy{{font-size:14px;line-height:1.9;max-width:700px}}
    .tag{{display:inline-block;border-radius:9px;padding:7px 11px;font-size:10px;font-weight:800;margin-top:14px}}
    .section-title{{font-family:'Manrope',sans-serif;font-size:20px;font-weight:800;margin-bottom:5px}}
    .section-copy{{font-size:12px;line-height:1.8;margin-bottom:15px}}
    .feature-card{{border-radius:18px;padding:22px;min-height:155px;margin-bottom:12px}}
    .feature-icon{{font-size:28px;margin-bottom:10px}}
    .feature-title{{font-family:'Manrope',sans-serif;font-size:20px;font-weight:800;margin-bottom:7px}}
    .feature-copy{{font-size:12px;line-height:1.8}}
    .panel{{border:1px solid var(--border);border-radius:17px;padding:19px;margin-bottom:16px}}
    .eyebrow-small{{font-size:10px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px}}
    .result-card{{border-radius:16px;padding:21px;margin:8px 0 18px}}
    .result-label{{font-size:10px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase}}
    .result-name{{font-family:'Manrope',sans-serif;font-size:clamp(24px,3vw,30px);font-weight:800;margin:8px 0;overflow-wrap:anywhere}}
    .result-note{{font-size:12px;line-height:1.8}}
    .score-row{{display:flex;justify-content:space-between;gap:12px;margin:14px 0 7px;font-size:13px;font-weight:600}}
    .score-track{{width:100%;height:7px;background:#9389A5;border-radius:99px;overflow:hidden}}
    .score-fill{{height:100%;background:linear-gradient(90deg,#8657F5,#F5E642);border-radius:99px}}
    .empty-state{{border-radius:16px;padding:26px 18px;text-align:center}}
    .empty-title{{font-family:'Manrope',sans-serif;font-weight:800;font-size:17px;margin:8px 0}}
    .empty-copy{{font-size:12px;line-height:1.8}}
    .notice{{border-radius:12px;padding:14px 16px;font-size:12px;line-height:1.8;margin-top:15px}}
    .footer{{border-top:1px solid;padding-top:18px;margin-top:35px;font-size:11px;line-height:1.8;text-align:center}}
    div.stButton>button,div.stFormSubmitButton>button,div.stDownloadButton>button{{
      width:100%;min-height:46px;display:flex;justify-content:center;align-items:center;
      text-align:center;white-space:normal;overflow-wrap:anywhere;padding:10px 14px;
      border-radius:11px;font-size:14px;font-weight:800;line-height:1.35;box-shadow:none;
    }}
    div.stButton>button,div.stFormSubmitButton>button{{
      background:#F5E642!important;border:1px solid #C8B900!important;color:#17130B!important;
    }}
    div.stButton>button *,div.stFormSubmitButton>button *{{color:inherit!important;fill:currentColor!important;font-weight:800!important}}
    div.stButton>button[kind="secondary"],div.stFormSubmitButton>button[kind="secondary"]{{
      background:#5B21B6!important;border:1px solid #7C3AED!important;color:#fff!important;
    }}
    div.stButton>button[kind="secondary"] *,div.stFormSubmitButton>button[kind="secondary"] *{{color:#fff!important;fill:#fff!important}}
    div.stDownloadButton>button{{background:#5B21B6!important;border:1px solid #7C3AED!important;color:#fff!important}}
    div.stDownloadButton>button *{{color:#fff!important;fill:#fff!important}}
    div.stButton>button:hover,div.stFormSubmitButton>button:hover{{background:#E8D62F!important;border-color:#A99A00!important;color:#17130B!important}}
    div.stButton>button[kind="secondary"]:hover,div.stFormSubmitButton>button[kind="secondary"]:hover,div.stDownloadButton>button:hover{{background:#6D28D9!important;border-color:#A78BFA!important;color:#fff!important}}
    div.stButton>button:focus-visible,div.stFormSubmitButton>button:focus-visible,div.stDownloadButton>button:focus-visible{{outline:3px solid #B49AFF!important;outline-offset:3px!important}}
    div.stButton>button:disabled,div.stFormSubmitButton>button:disabled,div.stDownloadButton>button:disabled{{background:#52505A!important;border-color:#696672!important;color:#fff!important;opacity:1!important;cursor:not-allowed}}
    div.stButton>button:disabled *,div.stFormSubmitButton>button:disabled *,div.stDownloadButton>button:disabled *{{color:#fff!important;fill:#fff!important}}
    [data-testid="stFileUploader"]{{border-radius:14px;padding:12px}}
    [data-testid="stImage"] img{{border-radius:12px}}
    [data-testid="stMetric"]{{border-radius:14px;padding:14px}}
    [data-testid="stAlert"]{{border-radius:12px}}
    @media(max-width:768px){{
      .block-container{{padding:1rem .8rem 1.5rem}}
      .hero{{border-radius:16px}}
      .feature-card{{min-height:auto;padding:18px}}
      div.stButton>button,div.stFormSubmitButton>button,div.stDownloadButton>button{{min-height:44px;font-size:13px}}
    }}
    @media(prefers-reduced-motion:reduce){{*{{transition:none!important;animation:none!important}}}}
    {theme_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 5. MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path, modified_time_ns, device_name):
    """Load a compatible ResNet18 checkpoint once."""
    path = Path(checkpoint_path)
    device = torch.device(device_name)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model", "model_weights"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            state_dict = candidate
            break

    if state_dict is None:
        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise ValueError("Model weights were not found.")

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    classes = checkpoint.get(
        "classes",
        checkpoint.get("class_names", DEFAULT_CLASSES),
    )
    class_to_idx = checkpoint.get("class_to_idx")

    if isinstance(class_to_idx, dict) and class_to_idx:
        classes = [
            name
            for name, index in sorted(
                class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]

    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("Invalid model class names.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, list(classes), device


def get_model(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    return load_model(
        str(model_path),
        model_path.stat().st_mtime_ns,
        str(DEVICE),
    )

# ============================================================
# 6. IMAGE PREPROCESSING — BLUR FIX
# ============================================================

@st.cache_data(show_spinner=False, max_entries=8)
def preprocess_image_bytes(image_bytes, scan_type):
    """
    Return model tensor and PNG preview.
    Original image is displayed separately at its original resolution.
    """
    with Image.open(BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    # High-quality resizing for model input only.
    model_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    # PNG avoids additional JPEG compression.
    preview_buffer = BytesIO()
    model_image.save(
        preview_buffer,
        format="PNG",
        optimize=False,
    )
    preview_bytes = preview_buffer.getvalue()

    # Keep the normalization consistent with the lung training pipeline.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tensor = transform(model_image).unsqueeze(0)

    return tensor, preview_bytes


def predict_image(image_bytes, scan_type, model, classes, device):
    start = time.perf_counter()

    tensor, preview_bytes = preprocess_image_bytes(
        image_bytes,
        scan_type,
    )
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(torch.argmax(probabilities).item())
    scores = {
        classes[i]: float(probabilities[i].item())
        for i in range(len(classes))
    }

    return {
        "label": classes[index],
        "scores": scores,
        "preview_bytes": preview_bytes,
        "elapsed_seconds": time.perf_counter() - start,
    }

# ============================================================
# 7. PRESENTATION HELPERS
# ============================================================

def render_score(name, score):
    percentage = max(0.0, min(100.0, score * 100.0))
    safe_name = html.escape(str(name))

    st.markdown(
        f"""
        <div class="score-row">
            <span>{safe_name}</span>
            <span>{percentage:.2f}%</span>
        </div>
        <div class="score-track">
            <div class="score-fill" style="width:{percentage:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(eyebrow, title, copy, highlight=""):
    safe_eyebrow = html.escape(eyebrow)
    safe_title = html.escape(title)
    safe_copy = html.escape(copy)
    safe_highlight = html.escape(highlight)

    title_html = safe_title
    if safe_highlight:
        title_html += f"<br><span>{safe_highlight}</span>"

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{safe_eyebrow}</div>
            <div class="hero-title">{title_html}</div>
            <div class="hero-copy">{safe_copy}</div>
            <div class="tag">AI-ASSISTED RESEARCH</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 8. HOME PAGE
# ============================================================

if st.session_state.page == "Home":
    render_hero(
        "Medical imaging workspace",
        "Better workflows.",
        (
            "Upload a medical image in the relevant workspace to explore "
            "the output of a configured image-classification model."
        ),
        "Clearer insights.",
    )

    st.markdown(
        '<div class="section-title">Choose your workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-copy">Select a scan type to continue.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🫁</div>
                <div class="feature-title">Lung Scan</div>
                <div class="feature-copy">
                    Upload a chest X-ray, view the resized model input,
                    and inspect class-wise scores.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Open Lung Scan →",
            key="home_open_lung",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "Lung Scan"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🩻</div>
                <div class="feature-title">Breast Scan</div>
                <div class="feature-copy">
                    Open the mammography workspace once the compatible
                    breast classification model is configured.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Open Breast Scan →",
            key="home_open_breast",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "Breast Scan"
            st.rerun()

    st.markdown(
        """
        <div class="notice">
            <b>Important</b><br>
            Predictions are experimental model outputs, not medical
            diagnoses. Clinical decisions require qualified professional
            evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 9. SCAN WORKSPACES
# ============================================================

else:
    scan_type = st.session_state.page
    is_lung = scan_type == "Lung Scan"
    model_path = LUNG_MODEL_PATH if is_lung else BREAST_MODEL_PATH

    title = (
        "Lung X-ray Analysis"
        if is_lung
        else "Breast Mammography Analysis"
    )
    description = (
        "Upload a chest X-ray to view its processed image and model scores."
        if is_lung
        else "Upload a mammogram to view its processed image and scores."
    )

    render_hero(scan_type, title, description)

    if not model_path.is_file():
        st.error(f"{scan_type} model checkpoint is not configured.")
        st.markdown("Expected model location:")
        st.code(str(model_path), language="text")

        if is_lung:
            st.info(
                "Check that lung training completed and the checkpoint "
                "exists at the path above."
            )
        else:
            st.info(
                "This breast checkpoint path is a placeholder. Configure "
                "the actual breast architecture, class order and training "
                "preprocessing before using breast predictions."
            )
        st.stop()

    try:
        with st.spinner("Preparing scan workspace..."):
            model, classes, model_device = get_model(model_path)
    except Exception as exc:
        st.error("The selected model could not be loaded.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.info(
            "Confirm that the checkpoint matches this ResNet18 architecture "
            "and has the expected classifier output dimensions."
        )
        st.stop()

    left, right = st.columns([1, 1], gap="large")

    state_prefix = scan_type.lower().replace(" ", "_")
    result_key = f"{state_prefix}_result"
    hash_key = f"{state_prefix}_hash"
    upload_key = f"upload_{state_prefix}"

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="eyebrow-small">Step 01</div>
                <div class="section-title">Upload scan</div>
                <div class="section-copy">
                    Choose a PNG, JPG or JPEG image.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Choose image",
            type=["png", "jpg", "jpeg"],
            key=upload_key,
            help="Choose a supported image file.",
        )

        image_bytes = None
        image_hash = None

        if uploaded is not None:
            try:
                image_bytes = uploaded.getvalue()
                image_hash = hashlib.sha256(image_bytes).hexdigest()

                with Image.open(BytesIO(image_bytes)) as opened:
                    width, height = opened.size

                    if width * height > MAX_IMAGE_PIXELS:
                        st.error(
                            "This image is too large to process safely. "
                            "Please upload a smaller image."
                        )
                        image_bytes = None
                        image_hash = None
                    else:
                        # Keep a correctly oriented, full-resolution image.
                        preview_image = ImageOps.exif_transpose(opened).convert("RGB")
                        dimensions = preview_image.size

                        # FIX: no thumbnail resizing or JPEG compression here.
                        st.image(
                            preview_image,
                            caption="Original uploaded scan",
                            use_container_width=True,
                        )
                        st.caption(
                            f"Original dimensions: "
                            f"{dimensions[0]} × {dimensions[1]}"
                        )

            except (UnidentifiedImageError, OSError, ValueError):
                image_bytes = None
                image_hash = None
                st.error("The selected file is not a valid image.")

        # Clear results when a different file is uploaded.
        if image_hash != st.session_state[hash_key]:
            st.session_state[hash_key] = image_hash
            st.session_state[result_key] = None

        if image_bytes is not None:
            with st.expander("View preprocessed image", expanded=False):
                try:
                    _, processed_bytes = preprocess_image_bytes(
                        image_bytes,
                        scan_type,
                    )
                    st.image(
                        processed_bytes,
                        caption="224 × 224 model input",
                        use_container_width=True,
                    )
                    st.caption(
                        "This preview is resized for model inference. "
                        "It is not the original-resolution image."
                    )
                    st.download_button(
                        "Download processed PNG",
                        data=processed_bytes,
                        file_name=f"{state_prefix}_224.png",
                        mime="image/png",
                        type="secondary",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning("Could not create the processed preview.")
                    with st.expander("Preprocessing details"):
                        st.code(str(exc))

        with st.form(f"analyze_form_{state_prefix}"):
            submitted = st.form_submit_button(
                "✦ Analyze Scan",
                type="primary",
                use_container_width=True,
                disabled=image_bytes is None,
            )

        if submitted and image_bytes is not None:
            try:
                with st.spinner("Analyzing image. Please wait..."):
                    result = predict_image(
                        image_bytes,
                        scan_type,
                        model,
                        classes,
                        model_device,
                    )
                result["image_hash"] = image_hash
                st.session_state[result_key] = result
            except Exception as exc:
                st.session_state[result_key] = None
                st.error(
                    "Analysis could not be completed. Check the image "
                    "and model configuration, then try again."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))

        if st.button(
            "Reset scan",
            key=f"reset_{state_prefix}",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state[result_key] = None
            st.session_state[hash_key] = None
            st.rerun()

    # ========================================================
    # RESULTS PANEL
    # ========================================================

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="eyebrow-small">Step 02</div>
                <div class="section-title">Scan results</div>
                <div class="section-copy">
                    Review the classification output and class-wise scores.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result = st.session_state[result_key]

        if result is not None and result.get("image_hash") != image_hash:
            result = None
            st.session_state[result_key] = None

        if result is None:
            st.markdown(
                """
                <div class="empty-state">
                    <div style="font-size:32px">✦</div>
                    <div class="empty-title">Ready for analysis</div>
                    <div class="empty-copy">
                        Upload a scan and select Analyze Scan to display
                        the model output here.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            label = str(result["label"])
            scores = result["scores"]
            confidence = scores[label]

            card_class = (
                "result-cancer"
                if label.casefold() == "cancer"
                else "result-normal"
            )

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="result-label">Model prediction</div>
                    <div class="result-name">{html.escape(label)}</div>
                    <div class="result-note">
                        Model score: {confidence * 100:.2f}%<br>
                        This is an experimental classification output,
                        not a medical diagnosis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric("Top class score", f"{confidence * 100:.2f}%")

            with metric2:
                st.metric(
                    "Inference time",
                    f'{result["elapsed_seconds"]:.2f} s',
                )

            st.markdown(
                '<div class="section-title">Class-wise scores</div>',
                unsafe_allow_html=True,
            )

            for class_name in classes:
                render_score(class_name, scores[class_name])

            summary = [
                f"CancerScan AI - {scan_type}",
                f"Predicted class: {label}",
                f"Top class score: {confidence * 100:.2f}%",
                f'Inference time: {result["elapsed_seconds"]:.2f} seconds',
                "",
                "Class-wise scores:",
            ]

            summary.extend(
                f"{name}: {score * 100:.2f}%"
                for name, score in scores.items()
            )

            summary.extend([
                "",
                "Research use only. Not a clinical diagnosis.",
            ])

            st.download_button(
                "Download result summary",
                data="\n".join(summary),
                file_name=f"{state_prefix}_summary.txt",
                mime="text/plain",
                type="secondary",
                use_container_width=True,
            )

            st.markdown(
                """
                <div class="notice">
                    <b>Research disclaimer</b><br>
                    This model has not been established as a clinically
                    validated diagnostic system. Softmax scores may not
                    represent calibrated clinical probabilities. Do not
                    use this result to diagnose, rule out or treat disease.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# 10. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>CancerScan AI</b> · Medical imaging research workspace<br><br>
        Yellow · Black · Purple · Not intended for clinical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
