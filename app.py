import torch
import os
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AI Surface Defect Detection")
st.caption("PatchCore anomaly inspection | AI-powered surface inspection")


# --------------------------------------------------
# Download model checkpoint from private Hugging Face
# --------------------------------------------------
@st.cache_resource(show_spinner="Downloading PatchCore model...")
def download_checkpoint():
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add your Hugging Face read token "
            "in Streamlit Cloud → App settings → Secrets."
        )

    return hf_hub_download(
        repo_id="satyam17-17/surface-defect-patchcore",
        filename="model.ckpt",
        token=token,
    )


try:
    downloaded_checkpoint = Path(download_checkpoint())

    @st.cache_resource(show_spinner="Preparing CPU-compatible checkpoint...")
    def prepare_cpu_checkpoint(checkpoint_path):
        cpu_path = Path(tempfile.gettempdir()) / "patchcore_cpu.ckpt"

        if not cpu_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            torch.save(checkpoint, cpu_path)

        return str(cpu_path)

    CHECKPOINT = Path(
        prepare_cpu_checkpoint(str(downloaded_checkpoint))
    )

except Exception as exc:
    st.error("Unable to prepare the PatchCore model.")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# Image upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a surface image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Unable to open the uploaded image: {exc}")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    if st.button("Inspect surface", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.png"
            image.save(input_path)

            try:
                with st.spinner(
                    "Loading PatchCore and inspecting image..."
                ):
                    model = Patchcore(
                        backbone="wide_resnet50_2",
                        pre_trained=True,
                    )

                    engine = Engine(
                        accelerator="cpu",
                        devices=1,
                    )
                    )

                    predictions = engine.predict(
                        model=model,
                        dataset=PredictDataset(path=input_path),
                        ckpt_path=str(CHECKPOINT),
                    )

                st.success("Inspection completed.")

                if not predictions:
                    st.warning(
                        "No prediction was returned by the model."
                    )
                    st.stop()

                prediction = predictions[0]

                # ------------------------------------------
                # Anomaly score
                # ------------------------------------------
                anomaly_score = getattr(
                    prediction, "pred_score", None
                )

                if anomaly_score is not None:
                    if hasattr(anomaly_score, "detach"):
                        score_array = (
                            anomaly_score.detach().cpu().numpy()
                        )
                    else:
                        score_array = np.asarray(anomaly_score)

                    score = float(score_array.reshape(-1)[0])

                    st.metric(
                        "Anomaly score",
                        f"{score:.4f}",
                    )

                    st.caption(
                        "Higher scores generally indicate more anomalous "
                        "images. This score is not a calibrated probability; "
                        "a validated threshold is needed for classification."
                    )

                # ------------------------------------------
                # Anomaly heatmap and overlay
                # ------------------------------------------
                anomaly_map = getattr(
                    prediction, "anomaly_map", None
                )

                pred_mask = getattr(
                    prediction, "pred_mask", None
                )

                col1, col2 = st.columns(2)

                if anomaly_map is not None:
                    if hasattr(anomaly_map, "detach"):
                        heat = (
                            anomaly_map.detach().cpu().numpy()
                        )
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
                            plt.get_cmap("jet")(heat_norm)[:, :, :3]
                            * 255
                        ).astype(np.uint8)

                        heat_rgb = Image.fromarray(colored_heat)

                        overlay = Image.blend(
                            image,
                            heat_rgb,
                            0.45,
                        )

                        with col1:
                            st.subheader("Anomaly heatmap")
                            st.image(
                                heat_rgb,
                                use_container_width=True,
                            )

                        with col2:
                            st.subheader("Heatmap overlay")
                            st.image(
                                overlay,
                                use_container_width=True,
                            )

                        st.caption(
                            "Heatmap colors show relative anomaly intensity "
                            "within this image, not a defect probability."
                        )
                    else:
                        with col1:
                            st.warning(
                                "The anomaly map does not have a valid 2D shape."
                            )
                else:
                    with col1:
                        st.info("No anomaly map was returned.")

                # ------------------------------------------
                # Predicted defect mask
                # ------------------------------------------
                st.subheader("Predicted defect mask")

                if pred_mask is not None:
                    if hasattr(pred_mask, "detach"):
                        mask = pred_mask.detach().cpu().numpy()
                    else:
                        mask = np.asarray(pred_mask)

                    mask = np.squeeze(mask)

                    if mask.ndim == 2 and mask.size > 0:
                        mask = np.nan_to_num(
                            mask.astype(np.float32),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
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
                            use_container_width=True,
                        )

                        if not np.any(binary_mask):
                            st.warning(
                                "The model returned an all-black mask. "
                                "No positive defect pixels were reported; "
                                "this does not prove the surface is defect-free."
                            )
                    else:
                        st.warning(
                            "The predicted mask does not have a valid 2D shape."
                        )
                else:
                    st.info("No predicted mask was returned.")

                st.info(
                    "This checkpoint was trained for scratch anomaly "
                    "detection. It does not identify every defect category."
                )

            except Exception as exc:
                st.error(f"Inspection failed: {exc}")
                st.exception(exc)

else:
    st.info("Upload an image to begin surface inspection.")
