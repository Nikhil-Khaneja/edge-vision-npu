# ImageNet-20 classification on a Mobilint NPU

A residual CNN trained from scratch on a 20-class ImageNet (ILSVRC-2012) subset,
then quantized to INT8 and run on a Mobilint ARIES NPU.

- **Data:** 20 classes × 300 training images, 20 × 50 validation images
- **I/O:** input `[240, 240, 3]`, output 20 logits
- **Params:** 2,763,060, no pretrained weights
- **Best validation accuracy (PyTorch, FP32):** 77.40%

## Architecture
Implemented as `ResNet20` in `model.py`. It is a compact residual network with
Kaiming initialization.

| Stage | Operation | Output |
|-------|-----------|--------|
| Stem    | Conv3×3 s2 + BN + ReLU | 120×120×32 |
| Layer 1 | BasicBlock(32→64, s2) + BasicBlock(64→64) | 60×60×64 |
| Layer 2 | BasicBlock(64→128, s2) + BasicBlock(128→128) | 30×30×128 |
| Layer 3 | BasicBlock(128→256, s2) + BasicBlock(256→256) | 15×15×256 |
| Head    | Global avg pool → Linear(256→20) | 20 logits |

Each BasicBlock has two 3×3 convs with BatchNorm. When the shape changes, the
shortcut is a 1×1 projection.

Design rationale:
- Residual blocks train stably on a small dataset in few epochs.
- Three ×2 downsampling stages shrink 240×240 to 15×15 before pooling. This keeps
  the model compact and friendly to quantization.
- Strong augmentation is the main regularizer for only 300 images per class.

## Training
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | SGD, Nesterov, momentum 0.9 |
| LR / schedule | 0.05, cosine annealing |
| Weight decay | 5e-4 |
| Batch / epochs | 64 / 40 |
| Loss | Cross-entropy |

**Train augmentation:** RandomResizedCrop(240, scale 0.7–1.0), horizontal flip,
ColorJitter(0.3, 0.3, 0.3, 0.1), ImageNet mean/std normalization.
**Validation:** Resize(256) → CenterCrop(240) → normalize.

```bash
python train.py                         # prints per-epoch train/val accuracy, saves checkpoint.pth
python eval.py                          # per-class + overall accuracy
python to_onnx.py                       # -> model.onnx, input [1,3,240,240], opset 13
python collect_calib_images.py          # balanced calibration subset from the TRAINING split
```
Or run everything in Colab with `colab_train.ipynb`.

## NPU deployment
1. `prepare_calib.py` builds Qubee calibration data. `resnet20_preproc.yaml` puts
   Resize → CenterCrop → Normalize → HWC inside the compiled model.
2. `compile_mxq.py` compiles ONNX to `model.mxq` (INT8).
3. `npu_infer.py` runs MACCEL inference on raw `uint8` HWC images and reports
   per-class and overall accuracy.

Steps 1–2 run inside the Qubee Docker container (`scripts/compile_in_docker.sh`).
Step 3 runs on the Windows NPU host (`scripts/npu_infer_windows.bat`). See
[`NPU_RUNBOOK.md`](NPU_RUNBOOK.md) for the exact commands.

## Data layout
Images are not included. The split files in `dataset/` list `<filename> <label>`.
Defaults expect the images here:
```
dataset/imagenet_train20a/<class>/<file>.JPEG
dataset/imagenet_val20/<file>.JPEG
```
