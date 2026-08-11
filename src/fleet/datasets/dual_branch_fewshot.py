"""Path-list dataset for few-shot training and evaluation (dual-branch)."""

from __future__ import annotations

import multiprocessing
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from fleet.utils import apply_fft_high_freq_mask


class DualBranchImageDataset(Dataset):
    """Dual-branch: DINOv3 224 + Xception 128 (FFT)."""

    def __init__(self, image_paths, labels, processor=None, xception_crop_size=128, is_training=True, max_load_warnings=5):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.xception_crop_size = xception_crop_size
        self.is_training = is_training
        self.max_load_warnings = max_load_warnings
        self._load_failure_count = multiprocessing.Value("i", 0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            with self._load_failure_count.get_lock():
                self._load_failure_count.value += 1
                failure_count = self._load_failure_count.value
            if failure_count <= self.max_load_warnings:
                print(f"Warning: failed to load image {img_path}: {e}")
                if failure_count == self.max_load_warnings:
                    print("Warning: further image load failures will not be printed individually")
            img = Image.new("RGB", (224, 224), color="black")

        width, height = img.size
        img_dinov3 = img.resize((224, 224), Image.BILINEAR)

        if self.processor:
            processed = self.processor(img_dinov3, return_tensors="pt")
            pixel_values_dinov3 = processed["pixel_values"].squeeze(0)
        else:
            tfm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            pixel_values_dinov3 = tfm(img_dinov3)

        if width < self.xception_crop_size or height < self.xception_crop_size:
            min_dim = min(width, height)
            scale = self.xception_crop_size / min_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.BILINEAR)
            width, height = new_width, new_height

        if self.is_training:
            left = random.randint(0, max(0, width - self.xception_crop_size))
            top = random.randint(0, max(0, height - self.xception_crop_size))
        else:
            left = (width - self.xception_crop_size) // 2
            top = (height - self.xception_crop_size) // 2

        img_crop = img.crop((left, top, left + self.xception_crop_size, top + self.xception_crop_size))
        img_tensor = transforms.ToTensor()(img_crop)
        pixel_values_xception = apply_fft_high_freq_mask(img_tensor, self.xception_crop_size)

        return pixel_values_dinov3, pixel_values_xception, label
