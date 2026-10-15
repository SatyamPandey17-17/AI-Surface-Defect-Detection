
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

PATCHCORE_CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "Patchcore"
    / "surface_scratch"
    / "latest"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

# Use the original NEU-trained YOLO model, not the fine-tuned one.
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

st.title("AI Surface Defect Detection")
st.caption(
    "Combined inspection using PatchCore anomaly detection "
    "and YOLO steel-defect classification"
)


# ============================================================
# LOAD MODELS ONCE AND CACHE THEM
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
        accelerator="auto",
        devices=1,
    )
    return model, engine


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not PATCHCORE_CHECKPOINT.exists():
    st.error(
        f"PatchCore checkpoint not found:\n{PATCHCORE_CHECKPOINT}"
    )
    st.stop()

if not YOLO_CHECKPOINT.exists():
    st.error(
        f"YOLO checkpoint not found:\n{YOLO_CHECKPOINT}\n\n"
        "Check that the original YOLO best.pt file exists."
    )
    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Models and supported defect classes"):
    st.write("**PatchCore:** Scratch-focused anomaly detection.")
    st.write(
        "**YOLO:** NEU steel-surface defect detector trained for "
        "six categories:"
    )
    st.write(
        "Crazing, inclusion, patches, pitted surface, "
        "rolled-in scale, and scratches."
    )
    st.warning(
        "Predictions are experimental, not production-validated. "
        "Performance may vary with image quality, lighting, material, "
        "and defect type."
    )


# ============================================================
# INPUT IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["png", "jpg", "jpeg", "bmp"],
)

confidence_threshold = st.slider(
    "YOLO detection confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05,
    help=(
        "Lower values show more predictions but can add false detections. "
        "Higher values show fewer, more confident predictions."
    ),
)


# ============================================================
# INSPECTION
# ============================================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open the uploaded image: {exc}")
        st.stop()

    st.subheader("Original image")
    st.image(image, use_container_width=True)

    if st.button("Inspect surface", type="primary"):

        # ----------------------------------------------------
        # YOLO DEFECT DETECTION
        # ----------------------------------------------------

        st.header("1. YOLO defect detection")

        try:
            with st.spinner("YOLO is detecting defect categories..."):
                yolo_model = load_yolo_model(YOLO_CHECKPOINT)

                yolo_device = 0 if torch.cuda.is_available() else "cpu"

                yolo_results = yolo_model.predict(
                    source=np.asarray(image),
                    imgsz=320,
                    conf=confidence_threshold,
                    device=yolo_device,
                    verbose=False,
                )

            result = yolo_results[0]
            annotated_image = result.plot()
            annotated_image = Image.fromarray(
                annotated_image[:, :, ::-1]
            )

            st.image(
                annotated_image,
                caption="YOLO predictions with bounding boxes",
                use_container_width=True,
            )

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                detection_rows = []

                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    coordinates = box.xyxy[0].tolist()

                    class_name = result.names.get(
                        class_id,
                        str(class_id),
                    )

                    detection_rows.append(
                        {
                            "Defect type": class_name,
                            "Confidence": f"{confidence * 100:.1f}%",
                            "X1": round(coordinates[0], 1),
                            "Y1": round(coordinates[1], 1),
                            "X2": round(coordinates[2], 1),
                            "Y2": round(coordinates[3], 1),
                        }
                    )

                st.success(
                    f"YOLO reported {len(detection_rows)} detection(s)."
                )
                st.dataframe(
                    detection_rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    "YOLO found no detections at the selected confidence "
                    "threshold. Try lowering the slider. No detections "
                    "does not prove the surface is defect-free."
                )

        except Exception as exc:
            st.error(f"YOLO detection failed: {exc}")
            st.exception(exc)

        # ----------------------------------------------------
        # PATCHCORE ANOMALY DETECTION
        # ----------------------------------------------------

        st.header("2. PatchCore anomaly inspection")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / "input.png"
                image.save(input_path)

                with st.spinner(
                    "PatchCore is calculating the anomaly score and heatmap..."
                ):
                    patchcore_model, engine = load_patchcore_model()

                    predictions = engine.predict(
                        model=patchcore_model,
                        dataset=PredictDataset(path=input_path),
                        ckpt_path=str(PATCHCORE_CHECKPOINT),
                    )

            if not predictions:
                st.warning(
                    "PatchCore did not return a prediction. "
                    "Check the terminal for additional errors."
                )
            else:
                prediction = predictions[0]

                anomaly_score = getattr(
                    prediction, "pred_score", None
                )
                anomaly_map = getattr(
                    prediction, "anomaly_map", None
                )
                pred_mask = getattr(
                    prediction, "pred_mask", None
                )

                if anomaly_score is not None:
                    score_array = np.asarray(
                        anomaly_score.detach().cpu().numpy()
                        if hasattr(anomaly_score, "detach")
                        else anomaly_score
                    )
                    score = float(score_array.reshape(-1)[0])

                    st.metric("PatchCore anomaly score", f"{score:.4f}")
                    st.caption(
                        "Higher scores generally indicate greater anomaly "
                        "intensity. This score is not a calibrated probability "
                        "and requires a validated decision threshold."
                    )

                # --------------------------------------------
                # ANOMALY HEATMAP AND OVERLAY
                # --------------------------------------------

                col1, col2 = st.columns(2)

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
                        )

                        heat_image = heat_image.resize(
                            image.size,
                            Image.Resampling.BILINEAR,
                        )

                        heat_norm = (
                            np.asarray(heat_image).astype(np.float32)
                            / 255.0
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

                        with col1:
                            st.subheader("Anomaly heatmap")
                            st.image(
                                heat_rgb,
                                use_container_width=True,
                            )
                            st.caption(
                                "Relative anomaly intensity within this "
                                "image; not a calibrated probability."
                            )

                        with col2:
                            st.subheader("Heatmap overlay")
                            st.image(
                                overlay,
                                use_container_width=True,
                            )
                    else:
                        st.warning(
                            "The anomaly map does not have a valid 2D shape."
                        )
                else:
                    st.info("PatchCore did not return an anomaly map.")

                # --------------------------------------------
                # PREDICTED MASK
                # --------------------------------------------

                st.subheader("PatchCore predicted defect mask")

                if pred_mask is not None:
                    if hasattr(pred_mask, "detach"):
                        mask = pred_mask.detach().cpu().numpy()
                    else:
                        mask = np.asarray(pred_mask)

                    mask = np.squeeze(mask)

                    if mask.ndim == 2 and mask.size > 0:
                        mask = np.nan_to_num(
                            mask.astype(np.float32)
                        )

                        binary_mask = (
                            (mask > 0).astype(np.uint8) * 255
                        )

                        mask_image = Image.fromarray(binary_mask)
                        mask_image = mask_image.resize(
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
                                "PatchCore returned an all-black mask. "
                                "This does not prove the surface is defect-free."
                            )
                    else:
                        st.warning(
                            "The predicted mask does not have a valid 2D shape."
                        )
                else:
                    st.info("PatchCore did not return a predicted mask.")

                st.info(
                    "PatchCore uses your scratch-focused checkpoint, while "
                    "YOLO predicts six NEU steel-defect categories. Their "
                    "outputs are separate analyses and are not combined into "
                    "one validated decision."
                )

        except Exception as exc:
            st.error(f"PatchCore inspection failed: {exc}")
            st.exception(exc)

else:
    st.info("Upload an image to begin surface inspection.")
