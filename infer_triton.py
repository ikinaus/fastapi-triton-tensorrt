import os
from io import BytesIO

import numpy as np
import tritonclient.grpc.aio as grpcclient
from PIL import Image
from tritonclient.utils import triton_to_np_dtype


class InferenceModule:
    def __init__(self) -> None:
        self.url = os.environ.get("TRITON_SERVER_URL", "127.0.0.1:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=self.url)

    @staticmethod
    def center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
        w, h = img.size
        left = (w - width) / 2
        top = (h - height) / 2
        right = (w + width) / 2
        bottom = (h + height) / 2

        return img.crop((left, top, right, bottom))

    def preprocess_image_pil(self, img: bytes) -> np.ndarray:
        pil_img = Image.open(BytesIO(img)).convert("RGB")

        resized_img = pil_img.resize((256, 256), 2)
        cropped_img = self.center_crop(resized_img, 224, 224)

        np_img = np.array(cropped_img).astype(np.float32)  # [224, 224, 3]

        normalizerd_img = (np_img / 225 - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )

        ordered_img = np.transpose(normalizerd_img, (2, 0, 1))  # [3, 224, 224]
        batched_img = np.expand_dims(ordered_img, axis=0)  # [1, 3, 224, 224]

        return batched_img
