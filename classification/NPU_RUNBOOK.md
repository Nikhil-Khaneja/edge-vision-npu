# Classification NPU runbook — Mobilint Qubee + MACCEL
# Project dir on the NPU machine: C:\NPU_Projects\classification
# (change this path in the two .bat files if you use something else).

All project files live in `C:\NPU_Projects\classification` on the NPU machine:

- `model.onnx`, `calib_images.zip`       (from Colab)
- `prepare_calib.py`, `compile_mxq.py`, `npu_infer.py`, `dataset.py`
- `resnet20_preproc.yaml`, `imagenet_train20.txt`, `imagenet_val20.txt`
- `imagenet_train20.zip`, `imagenet_val20.zip`   (unzipped data if needed)
- `qubee-0.12.0.0+aries2-py3-none-any.whl`, `maccel-0.30.1-cp312-cp312-win_amd64.whl`

---

## PART 1 — Compile ONNX -> MXQ (Docker)

### Step 1. Start Qubee container (CMD)
Easiest: double-click `C:\NPU_Projects\classification\scripts\npu_compile_docker.bat`
(host `C:/NPU_Projects/classification` is mounted as `/NPU_Projects`).

Or run manually:
```bat
docker run -it --name qb_compiler_container -v C:/NPU_Projects/classification:/NPU_Projects mobilint/qbcompiler:0.12-cpu-ubuntu22.04 /bin/bash
```
First run downloads the image (8–13 min).

### Step 2. Inside the container
```bash
cd /NPU_Projects
bash scripts/compile_in_docker.sh
```
This installs qubee, unzips data if needed, builds calibration, and
compiles `model.onnx` -> `model.mxq` (prints size as sanity check).

### Step 3. Exit Docker
```bash
exit
```

---

## PART 2 — NPU inference & accuracy (Windows)

### Step 4. Install MACCEL + run
Double-click `C:\NPU_Projects\classification\scripts\npu_infer_windows.bat`
(installs maccel wheel, then runs accuracy). Equivalent to:
```bat
cd C:\NPU_Projects\classification
python -m pip install "C:\NPU_Projects\classification\maccel-0.30.1-cp312-cp312-win_amd64.whl"
python npu_infer.py --mxq model.mxq --txt imagenet_val20.txt --dir imagenet_val20
```

### Step 5. Read per-class + overall accuracy (= leaderboard score basis).

---

## Pitfalls
- `prepare_calib.py` / `compile_mxq.py` need `qubee` -> Docker only.
- `npu_infer.py` needs `maccel` -> Windows only.
- ONNX input `[1,3,240,240]`, output 20 logits.
- If Qubee rejects the ONNX opset, re-export with `to_onnx.py`
  (opset 13, `dynamo=False`).
- If Python missing on Windows -> install via Microsoft Store.