import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


def throw_one(probability: float):
    return random.random() < probability


def image_resize(img, max_size=512):
    w, h = img.size

    if w >= h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    return img.resize((new_w, new_h))


def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size

    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1),
    }

    target_w, target_h = ratio_map[ratio]
    target_ratio = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    return image.crop(crop_box)


class CustomImageDataset(Dataset):

    def __init__(
        self,
        img_dir,
        img_size=512,
        caption_type="txt",
        random_ratio=False,
        caption_dropout_rate=0.1,
        cached_text_embeddings=None,
        cached_image_embeddings=None,
        control_dir=None,
        cached_image_embeddings_control=None,
    ):

        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate

        self.control_dir = control_dir

        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control

        print(f"Loaded {len(self.images)} training images.")

    def __len__(self):
        return len(self.images)

    def preprocess_image(self, img):

        if self.random_ratio:
            ratio = random.choice(
                ["16:9", "default", "1:1", "4:3"]
            )

            if ratio != "default":
                img = crop_to_aspect_ratio(img, ratio)

        img = image_resize(img, self.img_size)

        w, h = img.size

        img = img.resize(
            (
                (w // 32) * 32,
                (h // 32) * 32,
            )
        )

        img = torch.from_numpy(
            (np.array(img).astype(np.float32) / 127.5) - 1.0
        )

        img = img.permute(2, 0, 1)

        return img

    def __getitem__(self, idx):

        img_path = self.images[idx]
        filename = os.path.basename(img_path)

        try:

            ##################################################
            # Main Image
            ##################################################
            print("=" * 80)
            print("Cached image embedding keys:")
            print(list(self.cached_image_embeddings.keys()))
            print("Dataset filenames:")
            print([os.path.basename(x) for x in self.images])
            print("=" * 80)
            if self.cached_image_embeddings is None:
                print("NOOOOOOOOOOOOOOOOOO  self.cached_image_embeddings ------------------------------------------------- ")
                img = Image.open(img_path).convert("RGB")
                img = self.preprocess_image(img)

            else:
                print("YeSSSSSSSSSSSSSSSSSSSSSSSSSS self.cached_image_embeddings ------------------------------------------ ")
                img = self.cached_image_embeddings[filename]

            ##################################################
            # Control Image
            ##################################################

            if self.cached_control_image_embeddings is None:

                control_path = os.path.join(
                    self.control_dir,
                    filename,
                )

                if not os.path.exists(control_path):
                    raise FileNotFoundError(
                        f"Control image not found:\n{control_path}"
                    )

                control_img = Image.open(control_path).convert("RGB")
                control_img = self.preprocess_image(control_img)

            else:

                control_img = self.cached_control_image_embeddings[
                    filename
                ]

            ##################################################
            # Caption
            ##################################################

            txt_path = os.path.splitext(img_path)[0] + "." + self.caption_type

            if self.cached_text_embeddings is None:

                if not os.path.exists(txt_path):
                    raise FileNotFoundError(txt_path)

                with open(txt_path, encoding="utf-8") as f:
                    prompt = f.read()

                if throw_one(self.caption_dropout_rate):
                    prompt = " "

                return img, prompt, control_img

            else:

                txt_name = os.path.basename(txt_path)

                if throw_one(self.caption_dropout_rate):

                    emb = self.cached_text_embeddings[
                        txt_name + "empty_embedding"
                    ]

                else:

                    emb = self.cached_text_embeddings[
                        txt_name
                    ]

                return (
                    img,
                    emb["prompt_embeds"],
                    emb["prompt_embeds_mask"],
                    control_img,
                )

        except Exception as e:

            print("=" * 80)
            print("Error while loading sample")
            print("Image:", img_path)
            print("Error:", e)
            print("=" * 80)

            raise


def loader(train_batch_size, num_workers, **args):

    dataset = CustomImageDataset(**args)

    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    