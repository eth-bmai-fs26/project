import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256)):
        """
        root_dir/
            ├── original/
            └── segmented/
        """

        self.img_dir = os.path.join(root_dir, "original")
        self.mask_dir = os.path.join(root_dir, "segmented")

        valid_ext = (".png", ".jpg", ".jpeg")

        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if not f.startswith(".")
            and f.lower().endswith(valid_ext)
            and os.path.isfile(os.path.join(self.img_dir, f))
            and os.path.isfile(os.path.join(self.mask_dir, f))
        ])

        if len(self.files) == 0:
            raise RuntimeError("No valid image-mask pairs found.")

        self.img_transform = T.Compose([
            T.Grayscale(),
            T.Resize(image_size),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Grayscale(),
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, fname))
        mask = Image.open(os.path.join(self.mask_dir, fname))

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        # Binary mask: background = 0, object = 1
        mask = (mask > 0.5).float()

        return img, mask