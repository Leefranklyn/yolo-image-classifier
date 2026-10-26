# SCENE//CLASSIFIER

YOLOv8n-cls scene classifier (Intel image dataset — buildings / forest / glacier /
mountain / sea / street) served over a FastAPI app with an editorial-brutalist
blueprint interface.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Mounting the model

Inference runs on **onnxruntime** — no torch required. The export folder
`intel_v1_export/` (Kaggle Output zip, already extracted) supplies `best.onnx`.
Weight resolution order: `weights/*.onnx` → `intel_v1_export/*.onnx`
(`.pt` checkpoints are kept for reference only; converting them needs torch, which
this app does not install).

Restart the server — `/health` should report `{"model_state": "ready"}`. Until the
ONNX file exists, the app runs in a styled OFFLINE state and shows a fault panel
on upload attempts.

## API

| Route | Purpose |
|---|---|
| `GET /` | Upload UI + class manifest |
| `POST /predict` | multipart `file`, JSON body or request `Accept: application/json`. Returns `{html, data}` with the full 6-class probability table |
| `GET /health` | `{model_state, weights_file, n_classes}` |

Example:

```bash
curl -X POST -F file=scene.jpg -H "Accept: application/json" http://127.0.0.1:8000/predict
```
