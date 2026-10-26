import os
import shutil
import tempfile
from pathlib import Path

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
YOLO_MODEL_PATH = PROJECT_DIR / "best.pt"
YOLO_REPO_ID = HF_REPO_ID

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "bmp"]
MAX_BATCH_IMAGES = 6


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("Surface Defect Detection")
st.caption(
    "Steel surface inspection using PatchCore anomaly detection "
    "and YOLO defect detection."
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
    if PATCHCORE_CHECKPOINT.is_file() and PATCHCORE_CHECKPOINT.stat().st_size > 0:
        return str(PATCHCORE_CHECKPOINT)

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=PATCHCORE_FILENAME,
        repo_type="model",
        token=get_hf_token(),
    )

    temporary_path = PROJECT_DIR / "model_download.tmp"
    try:
        shutil.copyfile(downloaded_path, temporary_path)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("The downloaded PatchCore checkpoint is empty.")
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
        raise FileNotFoundError(f"PatchCore checkpoint not found: {source}")

    if (
        CPU_CHECKPOINT.is_file()
        and CPU_CHECKPOINT.stat().st_size > 0
        and CPU_CHECKPOINT.stat().st_mtime >= source.stat().st_mtime
    ):
        return str(CPU_CHECKPOINT)

    temporary_path = PROJECT_DIR / "model_cpu.tmp"
    try:
        checkpoint = torch.load(
            str(source),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        torch.save(checkpoint, str(temporary_path))
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("CPU checkpoint conversion produced an empty file.")
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
    if YOLO_MODEL_PATH.is_file() and YOLO_MODEL_PATH.stat().st_size > 0:
        return str(YOLO_MODEL_PATH)

    downloaded_path = hf_hub_download(
        repo_id=YOLO_REPO_ID,
        filename=YOLO_FILENAME,
        repo_type="model",
        token=get_hf_token(),
    )

    downloaded_file = Path(downloaded_path)
    if not downloaded_file.is_file() or downloaded_file.stat().st_size == 0:
        raise RuntimeError("The downloaded YOLO checkpoint is missing or empty.")

    return str(downloaded_file)


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
    engine = Engine(accelerator="cpu", devices=1)
    return model, engine


# ============================================================
# 8. MODEL INFORMATION
# ============================================================

with st.expander("Model status and supported defect classes", expanded=False):
    st.write("**PatchCore:** Anomaly detection using your trained checkpoint.")
    st.write(
        "**YOLO classes:** Crazing, inclusion, patches, pitted surface, "
        "rolled-in scale, and scratches."
    )
    st.caption(
        "YOLO predicts trained defect classes. PatchCore identifies unusual "
        "regions. Both can make mistakes and should be validated."
    )


# ============================================================
# 9. IMAGE SOURCE: PHONE CAMERA OR BATCH OF UP TO SIX PHOTOS
# ============================================================

st.header("Image acquisition")

mode = st.radio(
    "Choose inspection input",
    options=["Phone camera", "Batch photos (up to 6)"],
    horizontal=True,
    key="inspection_input_mode",
)

images_to_inspect = []

if mode == "Phone camera":
    st.info(
        "Open this website on your phone and tap Take photo. "
        "On supported mobile browsers, the phone camera will open."
    )
    camera_photo = st.camera_input(
        "Capture a product image",
        key="phone_camera_input",
    )
    if camera_photo is not None:
        try:
            captured_image = Image.open(camera_photo).convert("RGB")
            images_to_inspect = [("Phone camera capture", captured_image)]
            st.image(
                captured_image,
                caption="Captured product image",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Unable to open the captured image: {exc}")

else:
    selected_files = st.file_uploader(
        "Select up to 6 product images",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="batch_image_upload",
        help="Choose no more than six JPG, JPEG, PNG, or BMP images.",
    )

    if selected_files and len(selected_files) > MAX_BATCH_IMAGES:
        st.error(
            f"You selected {len(selected_files)} images. "
            f"Please select no more than {MAX_BATCH_IMAGES}."
        )
    elif selected_files:
        for uploaded in selected_files:
            try:
                image = Image.open(uploaded).convert("RGB")
                images_to_inspect.append((uploaded.name, image))
            except Exception as exc:
                st.warning(f"Could not read {uploaded.name}: {exc}")

        if images_to_inspect:
            st.write(f"Selected {len(images_to_inspect)} image(s).")
            preview_cols = st.columns(min(3, len(images_to_inspect)))
            for index, (name, image) in enumerate(images_to_inspect):
                with preview_cols[index % len(preview_cols)]:
                    st.image(image, caption=name, use_container_width=True)


# ============================================================
# 10. INSPECTION SETTINGS
# ============================================================

confidence_threshold = st.slider(
    "YOLO confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05,
    help="Lower thresholds may detect more objects but can increase false positives.",
)

run_inspection = st.button(
    "Inspect image(s)",
    type="primary",
    disabled=not images_to_inspect,
)

if not images_to_inspect:
    st.info("Capture a photo or select up to six photos to begin inspection.")

if not run_inspection:
    st.stop()


# ============================================================
# 11. LOAD MODELS ONCE FOR THE SELECTED BATCH
# ============================================================

yolo_model = None
patchcore_model = None
engine = None
cpu_checkpoint_path = None

with st.spinner("Loading AI models. The first run may take a while..."):
    try:
        yolo_checkpoint = ensure_yolo_checkpoint()
        yolo_model = load_yolo_model(yolo_checkpoint)
    except Exception as exc:
        st.error(f"YOLO model could not be loaded: {exc}")

    try:
        checkpoint_path = ensure_patchcore_checkpoint()
        cpu_checkpoint_path = make_cpu_checkpoint(checkpoint_path)
        patchcore_model, engine = load_patchcore_model()
    except Exception as exc:
        st.error(f"PatchCore model could not be prepared: {exc}")


# ============================================================
# 12. INSPECT EACH IMAGE
# ============================================================

if yolo_model is None and (patchcore_model is None or engine is None):
    st.error("Neither model could be loaded, so no inspection can be performed.")
    st.stop()

st.header("Inspection results")
all_summaries = []

# IMPORTANT:
# PatchCore anomaly scores are not probabilities. This optional threshold
# must be calibrated on a labelled validation set before using it to
# make real PASS/FAIL decisions.
patchcore_threshold = st.number_input(
    "PatchCore anomaly threshold (calibrate on validation data)",
    min_value=0.0,
    value=0.5,
    step=0.05,
    help=(
        "Placeholder threshold only. Calibrate this using known good and "
        "defective validation images before relying on PASS/FAIL."
    ),
)

for image_index, (image_name, image) in enumerate(images_to_inspect, start=1):
    st.divider()
    st.subheader(f"Image {image_index}: {image_name}")

    col_original, col_yolo = st.columns(2)
    with col_original:
        st.image(image, caption="Input image", use_container_width=True)

    yolo_has_defect = None
    yolo_error = None
    patchcore_score = None
    patchcore_error = None

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------
    if yolo_model is not None:
        try:
            with st.spinner(f"Running YOLO on image {image_index}..."):
                yolo_results = yolo_model.predict(
                    source=np.asarray(image),
                    imgsz=320,
                    conf=confidence_threshold,
                    device="cpu",
                    verbose=False,
                )

            result = yolo_results[0]
            annotated = result.plot()
            annotated_rgb = Image.fromarray(annotated[:, :, ::-1].copy())

            boxes = result.boxes
            yolo_has_defect = boxes is not None and len(boxes) > 0

            with col_yolo:
                st.image(
                    annotated_rgb,
                    caption="YOLO predictions",
                    use_container_width=True,
                )

            if yolo_has_defect:
                rows = []
                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    rows.append(
                        {
                            "Defect type": result.names.get(class_id, str(class_id)),
                            "Confidence": f"{confidence * 100:.1f}%",
                            "X1": round(x1, 1),
                            "Y1": round(y1, 1),
                            "X2": round(x2, 1),
                            "Y2": round(y2, 1),
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "YOLO found no boxes above the selected confidence threshold. "
                    "This alone does not prove the product is defect-free."
                )

        except Exception as exc:
            yolo_error = str(exc)
            with col_yolo:
                st.error(f"YOLO inference failed: {exc}")

    # --------------------------------------------------------
    # PatchCore inference
    # --------------------------------------------------------
    if patchcore_model is not None and engine is not None and cpu_checkpoint_path:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / f"input_{image_index}.png"
                image.save(input_path, format="PNG")

                with st.spinner(f"Running PatchCore on image {image_index}..."):
                    predictions = engine.predict(
                        model=patchcore_model,
                        dataset=PredictDataset(path=input_path),
                        ckpt_path=cpu_checkpoint_path,
                    )

            if predictions is None or len(predictions) == 0:
                raise RuntimeError("PatchCore returned no predictions.")

            prediction = predictions[0]
            anomaly_score = getattr(prediction, "pred_score", None)
            anomaly_map = getattr(prediction, "anomaly_map", None)

            if anomaly_score is not None:
                if hasattr(anomaly_score, "detach"):
                    score_values = anomaly_score.detach().cpu().numpy()
                else:
                    score_values = np.asarray(anomaly_score)

                score_values = np.nan_to_num(
                    score_values.astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                patchcore_score = float(score_values.reshape(-1)[0])
                st.metric("PatchCore anomaly score", f"{patchcore_score:.4f}")

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
                    low, high = float(heat.min()), float(heat.max())
                    heat_norm = (heat - low) / (high - low) if high > low else np.zeros_like(heat)
                    heat_image = Image.fromarray(
                        (heat_norm * 255).astype(np.uint8)
                    ).resize(image.size, Image.Resampling.BILINEAR)
                    heat_norm = np.asarray(heat_image).astype(np.float32) / 255.0
                    colored_heat = (
                        plt.get_cmap("jet")(heat_norm)[:, :, :3] * 255
                    ).astype(np.uint8)
                    heat_rgb = Image.fromarray(colored_heat)
                    overlay = Image.blend(image.convert("RGB"), heat_rgb, 0.45)

                    heat_col1, heat_col2 = st.columns(2)
                    with heat_col1:
                        st.image(heat_rgb, caption="Anomaly heatmap", use_container_width=True)
                    with heat_col2:
                        st.image(overlay, caption="Heatmap overlay", use_container_width=True)
                else:
                    st.warning("PatchCore returned an invalid anomaly-map shape.")

        except Exception as exc:
            patchcore_error = str(exc)
            st.error(f"PatchCore inference failed: {exc}")

    # --------------------------------------------------------
    # Per-image PASS/FAIL summary
    # --------------------------------------------------------
    # Defect is detected if YOLO finds a box OR PatchCore exceeds the
    # configured threshold. Since the PatchCore threshold is a placeholder,
    # do not treat this decision as production-ready until calibrated.
    if yolo_error is None and patchcore_error is None:
        model_signals = []
        if yolo_has_defect is not None:
            model_signals.append(bool(yolo_has_defect))
        if patchcore_score is not None:
            model_signals.append(patchcore_score >= patchcore_threshold)

        if not model_signals:
            verdict = "NOT INSPECTED"
        elif any(model_signals):
            verdict = "FAIL"
        else:
            verdict = "PASS"

    else:
        verdict = "INCOMPLETE — CHECK MODEL ERRORS"

    if verdict == "PASS":
        st.success(f"PASS — {image_name}")
    elif verdict == "FAIL":
        st.error(f"FAIL — {image_name}")
    else:
        st.warning(f"{verdict}: {image_name}")

    all_summaries.append(
        {
            "Image": image_name,
            "YOLO defect detected": (
                "Yes" if yolo_has_defect is True
                else "No" if yolo_has_defect is False
                else "Unavailable"
            ),
            "PatchCore score": (
                round(patchcore_score, 4)
                if patchcore_score is not None
                else "Unavailable"
            ),
            "Result": verdict,
        }
    )


# ============================================================
# 13. BATCH SUMMARY
# ============================================================

st.divider()
st.header("Batch summary")
st.dataframe(all_summaries, use_container_width=True, hide_index=True)

pass_count = sum(row["Result"] == "PASS" for row in all_summaries)
fail_count = sum(row["Result"] == "FAIL" for row in all_summaries)
incomplete_count = len(all_summaries) - pass_count - fail_count

col1, col2, col3 = st.columns(3)
col1.metric("PASS", pass_count)
col2.metric("FAIL", fail_count)
col3.metric("Incomplete", incomplete_count)

st.caption(
    "PASS/FAIL uses YOLO detection and the configured PatchCore score threshold. "
    "The threshold shown above is a placeholder until calibrated on labelled "
    "validation images. Validate the complete decision rule before relying on it."
)
