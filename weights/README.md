# MODEL MOUNT POINT

Drop trained weights here. Resolution order on app startup:

1. `best.pt` — ultralytics checkpoint (preferred; carries class names internally)
2. any `*.onnx` in this directory

Source: Kaggle notebook output (`/kaggle/working/intel_v1_export.zip`) — download from
the notebook's **Output** tab, unzip, and place `best.pt` in this directory.
Files here are gitignored.
