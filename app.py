from pathlib import Path
import os
import shutil
import tempfile

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ============================================================
# 1. PATHS AND MODEL CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

HF_REPO_ID = "satyam17-17/surface-defect-patchcore"

PATCHCORE_FILENAME = "model.ckpt"
YOLO_FILENAME = "best.pt"

PATCHCORE_CHECKPOINT = PROJECT_DIR / "model.ckpt"
CPU_CHECKPOINT = PROJECT_DIR / "model_cpu.ckpt"

# Optional local YOLO checkpoint.
YOLO_MODEL_PATH = PROJECT_DIR / "best.pt"

# Hugging Face repository containing both model files.
YOLO_REPO_ID = HF_REPO_ID


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AI Surface Defect Detection")

st.caption(
    "AI-powered steel surface inspection using "
    "PatchCore anomaly detection and YOLO defect detection."
)


# ============================================================
# 3. HUGGING FACE TOKEN
# ============================================================

def get_hf_token():
    """Read the optional Hugging Face token."""

    try:
        token = st.secrets.get("HF_TOKEN", None)
        if token:
            return token
    except Exception:
        pass

    return os.environ.get("HF_TOKEN")


# ============================================================
# 4. DOWNLOAD PATCHCORE CHECKPOINT
# ============================================================

@st.cache_resource(show_spinner=False)
def ensure_patchcore_checkpoint():

    if (
        PATCHCORE_CHECKPOINT.is_file()
        and PATCHCORE_CHECKPOINT.stat().st_size > 0
    ):
        return str(PATCHCORE_CHECKPOINT)

    token = get_hf_token()

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=PATCHCORE_FILENAME,
        repo_type="model",
        token=token,
    )

    temporary_path = PROJECT_DIR / "model_download.tmp"

    try:
        shutil.copyfile(downloaded_path, temporary_path)

        if (
            not temporary_path.is_file()
            or temporary_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "The downloaded PatchCore checkpoint is empty."
            )

        os.replace(temporary_path, PATCHCORE_CHECKPOINT)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return str(PATCHCORE_CHECKPOINT)


# ============================================================
# 5. PREPARE PATCHCORE CHECKPOINT FOR CPU
# ============================================================

@st.cache_resource(show_spinner=False)
def make_cpu_checkpoint(checkpoint_path):

    source = Path(checkpoint_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found: {source}"
        )

    if (
        CPU_CHECKPOINT.is_file()
        and CPU_CHECKPOINT.stat().st_size > 0
        and CPU_CHECKPOINT.stat().st_mtime >= source.stat().st_mtime
    ):
        return str(CPU_CHECKPOINT)

    temporary_path = PROJECT_DIR / "model_cpu.tmp"

    try:
        # Use only a checkpoint from your trusted model repository.
        checkpoint = torch.load(
            str(source),
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        torch.save(checkpoint, str(temporary_path))

        if (
            not temporary_path.is_file()
            or temporary_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "CPU checkpoint conversion produced an empty file."
            )

        os.replace(temporary_path, CPU_CHECKPOINT)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return str(CPU_CHECKPOINT)


# ============================================================
# 6. DOWNLOAD YOLO CHECKPOINT
# ============================================================

@st.cache_resource(show_spinner=False)
def ensure_yolo_checkpoint():

    # Use a local checkpoint if one has been included in deployment.
    if (
        YOLO_MODEL_PATH.is_file()
        and YOLO_MODEL_PATH.stat().st_size > 0
    ):
        return str(YOLO_MODEL_PATH)

    # Otherwise download best.pt from Hugging Face.
    token = get_hf_token()

    downloaded_path = hf_hub_download(
        repo_id=YOLO_REPO_ID,
        filename=YOLO_FILENAME,
        repo_type="model",
        token=token,
    )

    if not Path(downloaded_path).is_file():
        raise FileNotFoundError(
            "Hugging Face did not return a valid YOLO checkpoint."
        )

    if Path(downloaded_path).stat().st_size == 0:
        raise RuntimeError("The downloaded YOLO checkpoint is empty.")

    return str(downloaded_path)


# ============================================================
# 7. LOAD MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(str(model_path))


@st.cache_resource(show_spinner=False)
def load_patchcore_model():

    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    return model, engine


# ============================================================
# 8. MODEL INFORMATION
# ============================================================

with st.expander(
    "Model status and supported defect classes",
    expanded=False,
):

    st.write(
        "**PatchCore:** Scratch-focused anomaly detection."
    )

    st.write(
        "**YOLO classes:** Crazing, inclusion, patches, "
        "pitted surface, rolled-in scale, and scratches."
    )

    st.write(
        "**PatchCore checkpoint:** Downloaded from Hugging Face "
        "when needed."
    )

    st.write(
        "**YOLO checkpoint:** Downloaded from Hugging Face "
        "when needed."
    )

    st.caption(
        "The models work independently. PatchCore highlights unusual "
        "regions, while YOLO predicts categories learned during "
        "YOLO training."
    )


# ============================================================
# 9. IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["png", "jpg", "jpeg", "bmp"],
)

confidence_threshold = st.slider(
    "YOLO confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05,
)

if uploaded_file is None:
    st.info("Upload an image to begin inspection.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")

except Exception as exc:
    st.error(f"Unable to open the uploaded image: {exc}")
    st.stop()

st.subheader("Original image")

st.image(
    image,
    use_container_width=True,
)


# ============================================================
# 10. INSPECTION BUTTON
# ============================================================

if not st.button("Inspect surface", type="primary"):
    st.stop()


# ============================================================
# 11. YOLO DEFECT DETECTION
# ============================================================

st.header("1. YOLO defect detection")

try:

    with st.spinner("Downloading and loading YOLO model..."):

        yolo_checkpoint = ensure_yolo_checkpoint()
        yolo_model = load_yolo_model(yolo_checkpoint)

    with st.spinner("Running YOLO defect detection..."):

        yolo_results = yolo_model.predict(
            source=np.asarray(image),
            imgsz=320,
            conf=confidence_threshold,
            device="cpu",
            verbose=False,
        )

    result = yolo_results[0]

    # Ultralytics plot output is BGR. Convert to RGB.
    annotated = result.plot()
    annotated = Image.fromarray(
        annotated[:, :, ::-1].copy()
    )

    st.image(
        annotated,
        caption="YOLO bounding-box predictions",
        use_container_width=True,
    )

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        rows = []

        for box in boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            rows.append(
                {
                    "Defect type": result.names.get(
                        class_id,
                        str(class_id),
                    ),
                    "Confidence": f"{confidence * 100:.1f}%",
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1),
                }
            )

        st.success(
            f"YOLO detected {len(rows)} object(s)."
        )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "YOLO did not detect a defect at the selected confidence "
            "threshold. This does not prove the surface is defect-free."
        )

except Exception as exc:

    st.error(
        f"YOLO model loading or detection failed: {exc}"
    )

    with st.expander("YOLO technical error"):
        st.exception(exc)


# ============================================================
# 12. PATCHCORE ANOMALY INSPECTION
# ============================================================

st.header("2. PatchCore anomaly inspection")

try:

    # --------------------------------------------------------
    # Download the trained checkpoint.
    # --------------------------------------------------------

    with st.spinner(
        "Checking and downloading PatchCore checkpoint..."
    ):

        checkpoint_path = ensure_patchcore_checkpoint()

    # --------------------------------------------------------
    # Prepare a CPU-mapped checkpoint.
    # --------------------------------------------------------

    with st.spinner(
        "Preparing CPU-compatible PatchCore checkpoint..."
    ):

        cpu_checkpoint_path = make_cpu_checkpoint(
            checkpoint_path
        )

    # --------------------------------------------------------
    # Run inference on the uploaded image.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = Path(temp_dir) / "input.png"

        image.save(input_path, format="PNG")

        with st.spinner(
            "Calculating anomaly score and heatmap..."
        ):

            patchcore_model, engine = load_patchcore_model()

            predictions = engine.predict(
                model=patchcore_model,
                dataset=PredictDataset(path=input_path),
                ckpt_path=cpu_checkpoint_path,
            )

    if predictions is None or len(predictions) == 0:

        st.warning("PatchCore returned no predictions.")
        st.stop()

    prediction = predictions[0]

    anomaly_score = getattr(
        prediction,
        "pred_score",
        None,
    )

    anomaly_map = getattr(
        prediction,
        "anomaly_map",
        None,
    )

    pred_mask = getattr(
        prediction,
        "pred_mask",
        None,
    )


    # ========================================================
    # 13. ANOMALY SCORE
    # ========================================================

    st.subheader("Anomaly score")

    if anomaly_score is not None:

        if hasattr(anomaly_score, "detach"):
            score_values = (
                anomaly_score.detach().cpu().numpy()
            )
        else:
            score_values = np.asarray(anomaly_score)

        score_values = np.nan_to_num(
            score_values.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        score = float(
            score_values.reshape(-1)[0]
        )

        st.metric(
            label="PatchCore anomaly score",
            value=f"{score:.4f}",
        )

        st.caption(
            "Higher scores generally indicate greater anomaly intensity. "
            "The score is not automatically a probability or a validated "
            "defect/no-defect decision."
        )

    else:

        st.warning(
            "PatchCore did not return an anomaly score."
        )


    # ========================================================
    # 14. ANOMALY HEATMAP AND OVERLAY
    # ========================================================

    st.subheader("Anomaly heatmap and overlay")

    if anomaly_map is not None:

        if hasattr(anomaly_map, "detach"):
            heat = anomaly_map.detach().cpu().numpy()
        else:
            heat = np.asarray(anomaly_map)

        heat = np.squeeze(heat)

        if heat.ndim == 2 and heat.size > 0:

            heat = np.nan_to_num(
                heat.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            low = float(heat.min())
            high = float(heat.max())

            if high > low:
                heat_norm = (heat - low) / (high - low)
            else:
                heat_norm = np.zeros_like(
                    heat,
                    dtype=np.float32,
                )

            # Resize heatmap to the original image dimensions.
            heat_image = Image.fromarray(
                (heat_norm * 255).astype(np.uint8)
            ).resize(
                image.size,
                Image.Resampling.BILINEAR,
            )

            heat_norm = (
                np.asarray(heat_image).astype(np.float32)
                / 255.0
            )

            colored_heat = (
                plt.get_cmap("jet")(heat_norm)[:, :, :3]
                * 255
            ).astype(np.uint8)

            heat_rgb = Image.fromarray(colored_heat)

            overlay = Image.blend(
                image.convert("RGB"),
                heat_rgb,
                0.45,
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    heat_rgb,
                    caption="Anomaly heatmap",
                    use_container_width=True,
                )

            with col2:

                st.image(
                    overlay,
                    caption="Heatmap overlay",
                    use_container_width=True,
                )

            st.caption(
                "Colors represent relative anomaly intensity within "
                "this image. The heatmap is normalized for visualization."
            )

        else:

            st.warning(
                "PatchCore returned an invalid anomaly-map shape."
            )

    else:

        st.info(
            "PatchCore did not return an anomaly map."
        )


   

    # ========================================================
    # 16. FINAL INSPECTION SUMMARY
    # ========================================================

    st.divider()

    st.subheader("Inspection summary")

    st.write(
        "- **PatchCore:** Highlights unusual surface regions using "
        "the trained anomaly-detection checkpoint."
    )

    st.write(
        "- **YOLO:** Detects and labels defects using its separately "
        "trained checkpoint."
    )

    st.write(
        "- **Important:** Both models can make mistakes. Their outputs "
        "should be validated against labelled test images before "
        "industrial use."
    )

except Exception as exc:

    st.error(
        "PatchCore inspection failed. Review the technical error below."
    )

    error_text = str(exc).lower()

    if "401" in error_text or "403" in error_text:

        st.warning(
            "Hugging Face access may be denied. Confirm that the model "
            "repository is public or configure HF_TOKEN in Streamlit "
            "Cloud Secrets."
        )

    elif "cuda" in error_text:

        st.warning(
            "The checkpoint may contain device-specific tensors, "
            "or a dependency may be attempting to use CUDA."
        )

    with st.expander("PatchCore technical error"):
        st.exception(exc). 
        
