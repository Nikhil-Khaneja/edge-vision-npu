import argparse

import maccel
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from dataset import ImgDataset


def preprocess(img_path: str) -> np.ndarray:
    """Resize + CenterCrop + transpose to HWC, keep raw uint8 pixels.

    Normalization and channel-order handling are baked into the MXQ at
    compile time (see resnet20_preproc.yaml), so the NPU receives raw
    image bytes in the required [240, 240, 3] (HWC) layout.
    """
    img = Image.open(img_path).convert("RGB")
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=[256], interpolation=InterpolationMode.BILINEAR)
    out = F.center_crop(out, output_size=[240, 240])
    out = np.transpose(out.numpy(), axes=[1, 2, 0])  # CHW -> HWC
    out = out.astype(np.uint8)
    return out


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="NPU inference with MACCEL")
    parser.add_argument("--mxq", default=os.path.join(ROOT, "model.mxq"))
    parser.add_argument("--txt", default=os.path.join(ROOT, "dataset/imagenet_val20.txt"))
    parser.add_argument("--dir", default=os.path.join(ROOT, "dataset/imagenet_val20"))
    args = parser.parse_args()

    # ---- Load the compiled MXQ model on the NPU
    acc = maccel.Accelerator()
    model = maccel.Model(args.mxq)
    model.launch(acc)

    ds = ImgDataset(args.txt, args.dir, transform=None)

    class_correct = {c: 0 for c in range(20)}
    class_total = {c: 0 for c in range(20)}

    for idx, (filename, label) in enumerate(ds.samples):
        path = ds._resolve_path(filename)
        image = preprocess(path)
        print(f"[{idx+1}/{len(ds.samples)}] {filename} (label {label})")

        output = model.infer(image)
        if output is None:
            raise RuntimeError("Model inference returned no output.")
        out = np.asarray(output[0]).reshape(-1).astype(np.float32)
        pred = int(np.argmax(out))

        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
        print(f"    -> predicted {pred}  {'OK' if pred == label else 'WRONG'}")

    print("\n=== Per-class accuracy (NPU) ===")
    overall_c = overall_t = 0
    for c in range(20):
        acc_c = class_correct[c] / max(class_total[c], 1)
        overall_c += class_correct[c]
        overall_t += class_total[c]
        print(f"class {c:2d}: {class_correct[c]}/{class_total[c]} = {acc_c*100:6.2f}%")
    print(f"Overall accuracy on NPU: {overall_c/max(overall_t,1)*100:.2f}%")


if __name__ == "__main__":
    main()