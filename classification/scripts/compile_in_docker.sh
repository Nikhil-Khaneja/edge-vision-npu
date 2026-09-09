#!/bin/bash
# ==== Run INSIDE the Qubee Docker container (/NPU_Projects) ====
# The host folder C:/NPU_Projects/classification is mounted to /NPU_Projects, so all your
# scripts/data live there. Change nothing in this file.
set -e

cd /NPU_Projects

echo "== [1/4] Install Qubee =="
pip install /NPU_Projects/qubee-0.12.0.0+aries2-py3-none-any.whl

echo "== [2/4] Unzip data/calibration if needed =="
[ -d imagenet_train20a ] || unzip -o imagenet_train20.zip
[ -d imagenet_val20 ]   || unzip -o imagenet_val20.zip
[ -f calib_images/calib_00000.JPEG ] || unzip -o calib_images.zip

echo "== [3/4] Build calibration data (training split) =="
python prepare_calib.py \
  --train-txt imagenet_train20.txt \
  --train-dir imagenet_train20a \
  --images-dir calib_images \
  --num-calib 128

echo "== [4/4] Compile ONNX -> MXQ =="
python compile_mxq.py --onnx model.onnx --calib calib_data --out model.mxq

echo "== Done. model.mxq is in /NPU_Projects (also C:\\NPU_Projects\\classification) =="
ls -la model.mxq