# AI-Based Surface Defect Detection

## Project Overview
This final-year project uses computer vision and deep learning to identify surface anomalies in industrial images. It uses PatchCore through Anomalib and a Streamlit web interface.

## Features
- Upload a surface image through a web interface.
- Generate an anomaly score.
- Visualize anomaly intensity using a heatmap.
- Display the predicted defect mask when available.
- Run locally on Windows with Python.

## Technology Stack
- Python
- Anomalib 2.7.0
- PatchCore
- Streamlit
- NumPy
- Matplotlib
- Pillow

## Project Workflow
1. Upload an industrial surface image.
2. The PatchCore model extracts image features and evaluates anomalous patterns.
3. The application displays an anomaly score and visualization.
4. The result can support further inspection by a human operator.

## Installation
Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the Application

```cmd
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Model Checkpoint
The trained PatchCore checkpoint is not included in this repository. Place the compatible checkpoint in the expected local project path before running inference:

`results/Patchcore/surface_scratch/latest/weights/lightning/model.ckpt`

The checkpoint path and model configuration must match the code.

## Current Limitations
- The current checkpoint is trained for scratch anomaly detection.
- Anomaly scores are not calibrated probabilities.
- Performance on different materials and image conditions requires separate validation.
- Predicted masks and heatmaps should be checked against labeled test images.

## Future Improvements
- Evaluate performance on labeled normal and defective images.
- Add validated decision thresholds and quantitative metrics.
- Test additional materials and defect types.
- Integrate the model with an industrial inspection or conveyor system.

## Disclaimer
This is an academic prototype. It is not a validated replacement for industrial quality-control procedures.
