@echo off
REM ==== Classification NPU - Windows side: install MACCEL + run inference accuracy ====
REM Run inside CMD on the NPU machine AFTER compilation (Part 1).
REM Project dir is C:\NPU_Projects\classification (change PROJECT_DIR below if needed).

set PROJECT_DIR=C:\NPU_Projects\classification

cd /d %PROJECT_DIR%

echo == Step 1: install MACCEL runtime ==
python -m pip install "%PROJECT_DIR%\maccel-0.30.1-cp312-cp312-win_amd64.whl"
python -m pip install numpy pillow

echo == Step 2: run NPU inference on validation set ==
python npu_infer.py --mxq model.mxq --txt imagenet_val20.txt --dir imagenet_val20

pause