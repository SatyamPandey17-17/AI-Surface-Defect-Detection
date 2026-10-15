
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
# PATHS AND MODEL CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

HF_REPO_ID = "satyam17-17/surface-defect-patchcore"
HF_FILENAME = "model.ckpt"

PATCHCORE_CHECKPOINT = PROJECT_DIR / "model.ckpt"

YOLO_CHECKPOINT = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "neu"
    / "yolo11n_neu"
    / "weights"
    / "best.pt"
)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="AI Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AI Surface Defect Detection")
st.caption(
    "PatchCore anomaly inspection + YOLO steel-defect classification"
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

def get_hf_token():
    """Read token from Streamlit Secrets, never from source code."""
    try:
        return st.secrets.get("HF_TOKEN", None)
    except Exception:
        return os.environ.get("HF_TOKEN")


# ============================================================
# DOWNLOAD PATCHCORE CHECKPOINT
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
        filename=HF_FILENAME,
        repo_type="model",
        token=token,
    )

    # Copy into the project folder for consistent checkpoint handling.
    temporary_path = PATCHCORE_CHECKPOINT.with_suffix(".ckpt.part")

    try:
        shutil.copyfile(downloaded_path, temporary_path)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("Downloaded checkpoint is empty.")

        os.replace(temporary_path, PATCHCORE_CHECKPOINT)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return str(PATCHCORE_CHECKPOINT)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(str(model_path))


@st.cache_resource
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
# MODEL STATUS
# ============================================================

with st.expander("Model status and supported classes", expanded=False):
    st.write("**PatchCore:** Scratch-focused anomaly detection.")
    st.write(
        "**YOLO classes:** Crazing, inclusion, patches, pitted surface, "
        "rolled-in scale, and scratches."
    )
    st.caption(
        "The models perform separate analyses. Scores and predictions "
        "require validation before industrial use."
    )


# ============================================================
# IMAGE INPUT
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
    st.error(f"Unable to open image: {exc}")
    st.stop()

st.subheader("Original image")
st.image(image, use_container_width=True)


# ============================================================
# INSPECT
# ============================================================

if not st.button("Inspect surface", type="primary"):
    st.stop()


# ============================================================
# 1. YOLO DETECTION
# ============================================================

st.header("1. YOLO defect detection")

if not YOLO_CHECKPOINT.is_file():
    st.warning(
        "YOLO checkpoint best.pt is missing from Streamlit Cloud. "
        "YOLO will remain unavailable until you host or upload that file."
    )

else:
    try:
        with st.spinner("Running YOLO detection..."):
            yolo_model = load_yolo_model(str(YOLO_CHECKPOINT))

            yolo_results = yolo_model.predict(
                source=np.asarray(image),
                imgsz=320,
                conf=confidence_threshold,
                device="cpu",
                verbose=False,
            )

        result = yolo_results[0]
        annotated = result.plot()

        # YOLO returns BGR; convert to RGB for display.
        annotated = Image.fromarray(annotated[:, :, ::-1].copy())

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

                rows.append({
                    "Defect type": result.names.get(
                        class_id, str(class_id)
                    ),
                    "Confidence": f"{confidence * 100:.1f}%",
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1),
                })

            st.success(f"YOLO detected {len(rows)} object(s).")
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning(
                "No YOLO detections at this confidence threshold. "
                "This does not prove the surface is defect-free."
            )

    except Exception as exc:
        st.error(f"YOLO detection failed: {exc}")
        st.exception(exc)


# ============================================================
# 2. PATCHCORE ANOMALY INSPECTION
# ============================================================

st.header("2. PatchCore anomaly inspection")

try:
    with st.spinner(
        "Loading PatchCore checkpoint (approximately 188 MB)..."
    ):
        checkpoint_path = ensure_patchcore_checkpoint()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.png"
        image.save(input_path)

        with st.spinner("Calculating anomaly score and heatmap..."):
            patchcore_model, engine = load_patchcore_model()

            predictions = engine.predict(
                model=patchcore_model,
                dataset=PredictDataset(path=input_path),
                ckpt_path=checkpoint_path,
            )

    if predictions is None or len(predictions) == 0:
        st.warning("PatchCore returned no predictions.")
        st.stop()

    prediction = predictions[0]

    anomaly_score = getattr(prediction, "pred_score", None)
    anomaly_map = getattr(prediction, "anomaly_map", None)
    pred_mask = getattr(prediction, "pred_mask", None)

    # --------------------------------------------------------
    # ANOMALY SCORE
    # --------------------------------------------------------

    if anomaly_score is not None:
        if hasattr(anomaly_score, "detach"):
            score_values = anomaly_score.detach().cpu().numpy()
        else:
            score_values = np.asarray(anomaly_score)

        score = float(score_values.reshape(-1)[0])

        st.metric("PatchCore anomaly score", f"{score:.4f}")
        st.caption(
            "Higher scores generally indicate greater anomaly intensity. "
            "This score is not a probability or a validated defect decision."
        )

    # --------------------------------------------------------
    # HEATMAP AND OVERLAY
    # --------------------------------------------------------

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
                heat_norm = np.zeros_like(heat)

            heat_image = Image.fromarray(
                (heat_norm * 255).astype(np.uint8)
            ).resize(image.size, Image.Resampling.BILINEAR)

            heat_norm = (
                np.asarray(heat_image).astype(np.float32) / 255.0
            )

            colored_heat = (
                plt.get_cmap("jet")(heat_norm)[:, :, :3] * 255
            ).astype(np.uint8)

            heat_rgb = Image.fromarray(colored_heat)

            overlay = Image.blend(
                image.convert("RGB"),
                heat_rgb,
                0.45,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Anomaly heatmap")
                st.image(heat_rgb, use_container_width=True)

            with col2:
                st.subheader("Heatmap overlay")
                st.image(overlay, use_container_width=True)

            st.caption(
                "Colors represent relative anomaly intensity within the image."
            )

        else:
            st.warning("PatchCore returned an invalid anomaly-map shape.")
    else:
        st.info("PatchCore did not return an anomaly map.")

    # --------------------------------------------------------
    # PREDICTED MASK
    # --------------------------------------------------------

    st.subheader("PatchCore predicted defect mask")

    if pred_mask is not None:
        if hasattr(pred_mask, "detach"):
            mask = pred_mask.detach().cpu().numpy()
        else:
            mask = np.asarray(pred_mask)

        mask = np.squeeze(mask)

        if mask.ndim == 2 and mask.size > 0:
            mask = np.nan_to_num(mask.astype(np.float32))

            binary_mask = (
                (mask > 0).astype(np.uint8) * 255
            )

            mask_image = Image.fromarray(binary_mask).resize(
                image.size,
                Image.Resampling.NEAREST,
            )

            st.image(
                mask_image,
                caption="Binary predicted anomaly mask",
                use_container_width=True,
            )

            if not np.any(binary_mask):
                st.warning(
                    "The predicted mask is all black. "
                    "This does not prove the surface is defect-free."
                )

        else:
            st.warning("PatchCore returned an invalid predicted-mask shape.")

    else:
        st.info("PatchCore did not return a predicted mask.")

    st.info(
        "PatchCore identifies anomalous regions using your scratch-focused "
        "checkpoint. YOLO identifies defect categories when its separate "
        "checkpoint is available. These outputs are not combined into a "
        "single validated decision."
    )

except Exception as exc:
    st.error(
        "PatchCore inspection failed. Check the details below. "
        "If the error is 401 or 403, verify Hugging Face repository access "
        "and configure HF_TOKEN in Streamlit Cloud Secrets if needed."
    )
    st.exception(exc)
