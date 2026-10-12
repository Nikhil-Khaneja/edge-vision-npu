# Image restoration on a Mobilint NPU

Recover a clean 256×256 image from a degraded one: blur from downscaling, colour
noise and JPEG blocking. The model is compiled to INT8 and scored by **mean PSNR
on the NPU**.

- **Data:** 2,217 training pairs, 110 validation pairs (`LR2/` → `HR2/`, 256×256,
  pixel-aligned)
- **Baseline:** returning the input unchanged scores **25.12 dB** on validation.
  The real challenge is to beat it by a clear margin *and* keep that gain after
  INT8 quantization.

## Architecture comparison
Each design was trained for the same 3,000 steps (L1 loss, 128 crops, batch 16,
EMA weights). Full log: [`results/compare.log`](results/compare.log).

| Arch | Params | Val PSNR (FP32) |
|------|--------|-----------------|
| `plain` (16 ResBlocks @ 256²) | 1.22M | 25.973 dB |
| **`dual` (DualScaleSR)** | **2.66M** | **26.066 dB** |
| `unet` (3-level Res-UNet) | 4.85M | 26.033 dB |

All three were still improving at 3k steps.

### DualScaleSR
```
x ─ head conv ─ 2 ResBlocks @256 ─┬──────────────────────────────┐
                                  └ stride-2 conv ─ 12 ResBlocks @128 (96 ch, long skip)
                                                    ─ conv + PixelShuffle ─┤ concat
                                       1×1 fuse ─ 2 ResBlocks ─ conv → 3 ch ┘
out = x + tail          (global residual)
```
- **No BatchNorm.** Each residual branch is scaled down at init and the final conv
  starts at zero, so the untrained network outputs its input exactly.
- **Only NPU-safe ops:** Conv, ReLU, Add, Concat, DepthToSpace.
- The 128×128 deep branch adds receptive field cheaply, and the 256×256 branch
  keeps fine detail.

## Usage
```bash
python model.py                                   # sanity check: params, output shape, identity at init
bash compare_archs.sh                             # reproduce the comparison table
python train.py --arch dual --steps 60000 --out runs/dual_full
python train.py --arch dual --steps 10000 --loss mse --lr 5e-5 --crop 256 \
    --init runs/dual_full/best.pth --out runs/dual_mse          # MSE fine-tune
python to_onnx.py                                 # export + ONNX Runtime vs PyTorch PSNR check
python collect_calib_images.py                    # 200 calibration images from the TRAINING split
```
Training is Adam-based, with warmup + cosine LR, gradient clipping and EMA weights.
Validation PSNR is computed after clipping and rounding to `uint8`, matching
the NPU output path.

## NPU deployment
- `sr_preproc.yaml` puts the `/255` scaling and HWC order inside the `.mxq`, so the
  NPU reads raw `uint8 [256,256,3]`.
- `prepare_calib.py` and `compile_mxq.py` run inside the Qubee Docker container.
  `compile_mxq.py` also accepts `--percentile` and `--mode` for different
  calibration strategies.
- `npu_infer.py` runs on the Windows NPU host and prints mean NPU PSNR. It
  detects CHW vs HWC output on its own.

See [`NPU_RUNBOOK.md`](NPU_RUNBOOK.md) for the exact commands.

## Status
- [x] Architecture comparison (FP32): `dual` chosen
- [x] ONNX export verified (ONNX Runtime PSNR = PyTorch PSNR = 26.066 dB)
- [ ] Compile + measure PSNR on the NPU (INT8)
- [ ] Long training run + MSE fine-tune
- [ ] Report the FP32 → INT8 PSNR gap

## Data layout
Data is not included. Expected layout:
```
data/train/LR2/*.jpg   data/train/HR2/*.jpg
data/valid/LR2/*.jpg   data/valid/HR2/*.jpg
```
