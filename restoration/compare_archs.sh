#!/bin/bash
# Train each candidate with the same short budget and print a PSNR table.
# ~25-40 min per model on an M-series Mac (MPS); much faster on a Colab GPU.
set -e
cd "$(dirname "$0")"
STEPS=${STEPS:-3000}

for arch in plain dual unet; do
  python3 train.py --arch "$arch" --steps "$STEPS" --eval-every 500 --out "runs/cmp_$arch"
done

python3 - <<'EOF'
import json
print(f"\n{'arch':6s} {'params':>8s} {'best val PSNR':>14s} {'minutes':>8s}")
for a in ["plain", "dual", "unet"]:
    h = json.load(open(f"runs/cmp_{a}/history.json"))
    print(f"{a:6s} {h['params_M']:7.2f}M {h['best_val_psnr']:13.3f}  {h['history'][-1]['minutes']:7.1f}")
EOF
