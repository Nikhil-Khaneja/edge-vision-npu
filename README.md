# edge-vision-npu

Computer-vision models built from scratch in PyTorch and run all the way to INT8
inference on a **Mobilint ARIES NPU**:

```
PyTorch training  ->  ONNX export  ->  Qubee INT8 calibration + compile (.mxq)  ->  MACCEL inference on NPU
```

| Project | Task | Model | Result (FP32) |
|---------|------|-------|---------------|
| [`classification/`](classification/) | 20-class ImageNet subset, 240×240 input | Residual CNN, no pretraining | in progress |

## Repository layout
```
classification/   ImageNet-20 classifier  (train, eval, ONNX, calibration, compile, NPU infer)
```
Each project folder has its own README and an `NPU_RUNBOOK.md` with the exact commands for the
NPU machine.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The NPU steps also need Mobilint's SDK, which is **not** on PyPI and not included here:
- `qubee` (compiler), run inside the `mobilint/qbcompiler` Docker image
- `maccel` (runtime), on the Windows machine attached to the NPU

## Data
Datasets are not included. See each project's README for the expected layout.
