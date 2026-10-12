# Restoration NPU runbook (Mobilint Qubee + MACCEL, Windows NPU machine)

Put this folder at `C:\NPU_Projects\restoration`, next to the SDK wheels in
`C:\NPU_Projects` (`qubee-0.12.0.0+aries2-py3-none-any.whl`,
`maccel-0.30.1-cp312-cp312-win_amd64.whl`).

## Part 1 — Compile ONNX -> MXQ (Docker)

```bat
docker run -it --name qb_restoration -v C:/NPU_Projects:/NPU_Projects mobilint/qbcompiler:0.12-cpu-ubuntu22.04 /bin/bash
```
(If the container already exists: `docker start -ai qb_restoration`.)

Inside the container:
```bash
pip install /NPU_Projects/qubee-0.12.0.0+aries2-py3-none-any.whl
cd /NPU_Projects/restoration
python prepare_calib.py --images-dir calib_images --calib-dir calib_data
python compile_mxq.py --onnx runs/cmp_dual/model.onnx --calib calib_data --out model.mxq
exit
```

## Part 2 — Run on the NPU (Windows CMD)

```bat
python -m pip install "C:\NPU_Projects\maccel-0.30.1-cp312-cp312-win_amd64.whl"
python -m pip install numpy pillow
cd C:\NPU_Projects\restoration
python npu_infer.py --mxq model.mxq --data data/valid
```

Compare the printed `Mean PSNR on NPU` with the ONNX number (26.066 dB).
Also check the printed raw output range on the first image (should be ~0..1).

## Trying other calibration settings
Re-run only the compile step with a different output name, then infer again:
```bash
python compile_mxq.py --onnx runs/cmp_dual/model.onnx --calib calib_data --out model_p999.mxq --percentile 0.999
python compile_mxq.py --onnx runs/cmp_dual/model.onnx --calib calib_data --out model_max.mxq --mode 0
```
