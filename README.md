# edge-vision-npu

Two computer-vision models, each built from scratch in PyTorch and run all the way
to INT8 inference on a **Mobilint ARIES NPU**:

```
PyTorch training  ->  ONNX export  ->  Qubee INT8 calibration + compile (.mxq)  ->  MACCEL inference on NPU
```

| Project | Task | Model | Result (FP32) |
|---------|------|-------|---------------|
| [`classification/`](classification/) | 20-class ImageNet subset, 240×240 input | Residual CNN, 2.76M params, no pretraining | 77.40% val accuracy |
| [`restoration/`](restoration/) | Restore degraded 256×256 images (blur, colour noise, JPEG blocking) | `DualScaleSR`, a two-scale residual network, 2.66M params | 26.07 dB val PSNR (input as-is: 25.12 dB) |

## Highlights
- **NPU-friendly by design.** The restoration models use only Conv, ReLU, Add, Concat and
  PixelShuffle. Preprocessing (resize / normalize / HWC) is built into the compiled
  `.mxq`, so the NPU reads raw `uint8` images.
- **Architecture study.** Three restoration designs (single-scale ResNet, dual-scale,
  Res-UNet) trained with the same budget and compared on validation PSNR
  ([`restoration/results/`](restoration/results/)).
- **Training details.** AdamW with warmup and cosine decay, EMA weights, a model that
  outputs its input unchanged before training, and a PSNR metric that rounds and clips
  outputs exactly like the NPU output path.
- **Checked deployment.** ONNX Runtime output is compared against PyTorch before compiling.

## Repository layout
```
classification/   ImageNet-20 classifier  (train, eval, ONNX, calibration, compile, NPU infer)
restoration/      Image restoration       (3 architectures, train, ONNX, calibration, compile, NPU infer)
```
Each folder has its own README and an `NPU_RUNBOOK.md` with the exact commands for the
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

## License
[MIT](LICENSE)
