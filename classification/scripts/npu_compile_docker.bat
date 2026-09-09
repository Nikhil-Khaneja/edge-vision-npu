@echo off
REM ==== Classification NPU - Docker side: install Qubee + compile ONNX -> MXQ ====
REM Run inside CMD on the NPU machine. Does NOT do inference.
REM Mounts HOST path C:/NPU_Projects/classification (this project dir) as /NPU_Projects.
REM If your dir differs, change HOST_DIR below.

set HOST_DIR=C:/NPU_Projects/classification

REM Start existing container if present, else create it.
docker start qb_compiler_container >nul 2>&1
if errorlevel 1 (
    docker run -it --name qb_compiler_container -v %HOST_DIR%:/NPU_Projects mobilint/qbcompiler:0.12-cpu-ubuntu22.04 /bin/bash
) else (
    docker exec -it qb_compiler_container /bin/bash
)

echo Done. Container exited. MXQ should be at C:\NPU_Projects\classification\model.mxq
pause