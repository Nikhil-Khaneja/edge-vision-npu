import os
from PIL import Image
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    """ImageNet-20 subset dataset.

    The training .txt maps each image to one of 20 labels. Images are grouped
    by class subfolder (e.g. imagenet_train20a/n01737021/n01737021_10059.JPEG)
    while the validation images live in a single flat folder. The filename
    embeds the WordNet class id, so we can locate either layout automatically.

    Args:
        txt_path: path to the label list ("<filename>  <label>")
        img_dir:  root directory holding the images
        transform: torchvision transform applied to every sample
    """

    def __init__(self, txt_path, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                filename = parts[0]
                label_str = parts[1]
                self.samples.append((filename, int(label_str)))

    def _resolve_path(self, filename):
        # 1) flat layout: img_dir/filename
        direct = os.path.join(self.img_dir, filename)
        if os.path.exists(direct):
            return direct
        # 2) class-subdir layout: img_dir/<class_id>/filename
        #    class id is everything before the last underscore
        base = filename.rsplit("_", 1)[0]
        sub = os.path.join(self.img_dir, base, filename)
        if os.path.exists(sub):
            return sub
        raise FileNotFoundError(
            f"Could not locate '{filename}' under '{self.img_dir}'"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = self._resolve_path(filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
