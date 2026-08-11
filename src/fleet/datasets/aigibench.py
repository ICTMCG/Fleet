"""Training data for the dual-branch pipeline under the AIGIBench directory layout."""

from __future__ import annotations

import glob
import multiprocessing
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from fleet.utils import apply_fft_high_freq_mask


def collect_aigibench_image_paths(data_dir):
    """Collect AIGIBench image paths, supporting both category/category/{0_real,1_fake} and category/{0_real,1_fake} layouts."""
    image_paths = []
    labels = []
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    for category_dir in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category_dir)
        if not os.path.isdir(category_path):
            continue

        candidates = [os.path.join(category_path, category_dir), category_path]
        actual_dir = None
        for cand in candidates:
            real_dir = os.path.join(cand, "0_real")
            fake_dir_fake = os.path.join(cand, "1_fake")
            fake_dir_false = os.path.join(cand, "1_false")
            if os.path.exists(real_dir) or os.path.exists(fake_dir_fake) or os.path.exists(fake_dir_false):
                actual_dir = cand
                break
        if actual_dir is None:
            continue

        real_dir = os.path.join(actual_dir, "0_real")
        fake_dir_fake = os.path.join(actual_dir, "1_fake")
        fake_dir_false = os.path.join(actual_dir, "1_false")
        fake_dir = fake_dir_fake if os.path.exists(fake_dir_fake) else fake_dir_false

        if os.path.exists(real_dir):
            tmp = []
            for ext in image_extensions:
                tmp.extend(glob.glob(os.path.join(real_dir, ext)))
                tmp.extend(glob.glob(os.path.join(real_dir, ext.upper())))
            image_paths.extend(tmp)
            labels.extend([1] * len(tmp))

        if os.path.exists(fake_dir):
            tmp = []
            for ext in image_extensions:
                tmp.extend(glob.glob(os.path.join(fake_dir, ext)))
                tmp.extend(glob.glob(os.path.join(fake_dir, ext.upper())))
            image_paths.extend(tmp)
            labels.extend([0] * len(tmp))

    return image_paths, labels


class AIGIBenchDataset(Dataset):
    """AIGIBench: category/category/0_real and 1_fake (or 1_false)."""

    def __init__(self, data_dir, processor=None, xception_crop_size=128, max_load_warnings=5):
        self.processor = processor
        self.xception_crop_size = xception_crop_size
        self.max_load_warnings = max_load_warnings
        self._load_failure_count = multiprocessing.Value("i", 0)
        self.image_paths = []
        self.labels = []

        self.image_paths, self.labels = collect_aigibench_image_paths(data_dir)

        fake_count = sum(1 for label in self.labels if label == 0)
        real_count = len(self.labels) - fake_count
        print(f"AIGIBench[{data_dir}]: total={len(self.image_paths)}, fake={fake_count}, real={real_count}")

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
            img = Image.new("RGB", (224, 224), (0, 0, 0))

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

        left = random.randint(0, max(0, width - self.xception_crop_size))
        top = random.randint(0, max(0, height - self.xception_crop_size))
        img_crop = img.crop((left, top, left + self.xception_crop_size, top + self.xception_crop_size))
        img_tensor = transforms.ToTensor()(img_crop)
        pixel_values_xception = apply_fft_high_freq_mask(img_tensor, self.xception_crop_size)

        return pixel_values_dinov3, pixel_values_xception, label
