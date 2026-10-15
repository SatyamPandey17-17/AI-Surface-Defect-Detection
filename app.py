
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "Patchcore"
    / "surface_scratch"
    / "latest"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

st.set_page_config(
    page_title="AI Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AI Surface Defect Detection")
st.caption("PatchCore anomaly inspection | Local Windows application")

if not CHECKPOINT.exists():
    st.error(f"Model checkpoint not found: {CHECKPOINT}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a surface image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    if st.button("Inspect surface", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.png"
            image.save(input_path)

            try:
                with st.spinner("Loading PatchCore and inspecting image..."):
                    model = Patchcore(
                        backbone="wide_resnet50_2",
                        pre_trained=True,
                    )

                    engine = Engine(
                        accelerator="auto",
                        devices=1,
                    )

                    predictions = engine.predict(
                        model=model,
                        dataset=PredictDataset(path=input_path),
                        ckpt_path=str(CHECKPOINT),
                    )

                st.success("Inspection completed.")

                if predictions:
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
                        score = float(anomaly_score.reshape(-1)[0])
                        st.metric("Anomaly score", f"{score:.4f}")
                        st.caption(
                            "Higher scores generally indicate more anomalous "
                            "images. A validated decision threshold is still needed."
                        )

                    col1, col2 = st.columns(2)

                    # Convert the anomaly map into a 2D NumPy array
                    heat = None

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

                            # Normalize for visualization only
                            low = float(heat.min())
                            high = float(heat.max())

                            if high > low:
                                heat_norm = (heat - low) / (high - low)
                            else:
                                heat_norm = np.zeros_like(heat)

                            # Resize the heatmap to the original image dimensions
                            heat_image = Image.fromarray(
                                (heat_norm * 255).astype(np.uint8)
                            )
                            heat_image = heat_image.resize(
                                image.size,
                                Image.Resampling.BILINEAR,
                            )

                            heat_norm = (
                                np.asarray(heat_image).astype(np.float32) / 255.0
                            )

                            # Apply a color map
                            colored_heat = (
                                plt.get_cmap("jet")(heat_norm)[:, :, :3] * 255
                            ).astype(np.uint8)

                            heat_rgb = Image.fromarray(colored_heat)

                            # Overlay the heatmap on the original image
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
                                    "Colors show relative anomaly intensity "
                                    "within this image, not a calibrated probability."
                                )

                            with col2:
                                st.subheader("Heatmap overlay")
                                st.image(
                                    overlay,
                                    use_container_width=True,
                                )
                        else:
                            with col1:
                                st.warning(
                                    "The anomaly map does not have a valid 2D shape."
                                )
                    else:
                        with col1:
                            st.info("No anomaly map was returned.")

                    # Display the model's predicted mask separately
                    st.subheader("Predicted defect mask")

                    if pred_mask is not None:
                        if hasattr(pred_mask, "detach"):
                            mask = pred_mask.detach().cpu().numpy()
                        else:
                            mask = np.asarray(pred_mask)

                        mask = np.squeeze(mask)

                        if mask.ndim == 2 and mask.size > 0:
                            mask = np.nan_to_num(mask.astype(np.float32))

                            # Convert the predicted mask to black and white
                            binary_mask = (mask > 0).astype(np.uint8) * 255
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
                                    "This means no positive defect pixels were "
                                    "reported by the model for this image. "
                                    "It does not prove the surface is defect-free."
                                )
                        else:
                            st.warning(
                                "The predicted mask does not have a valid 2D shape."
                            )
                    else:
                        st.info("No predicted mask was returned.")

                    st.info(
                        "This checkpoint is trained for scratch anomaly "
                        "detection. It does not identify every defect category."
                    )
                else:
                    st.warning(
                        "No prediction object was returned. Check the terminal output."
                    )

            except Exception as exc:
                st.error(f"Inspection failed: {exc}")
                st.exception(exc)
else:
    st.info("Upload an image to begin surface inspection.")
