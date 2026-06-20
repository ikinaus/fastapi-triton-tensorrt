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

        resized_img = pil_img.resize((256, 256), Image.Resampling.BILINEAR)
        cropped_img = self.center_crop(resized_img, 224, 224)

        np_img = np.array(cropped_img).astype(np.float32)  # [224, 224, 3]

        normalized_img = (np_img / 255 - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )

        ordered_img = np.transpose(normalized_img, (2, 0, 1))  # [3, 224, 224]
        batched_img = np.expand_dims(ordered_img, axis=0)  # [1, 3, 224, 224]

        return batched_img

    async def infer_image(
        self, img: bytes, model_name: str = "classifier_onnx"
    ) -> dict:

        model_meta = await self.triton_client.get_model_metadata(model_name)

        assert model_meta is not None, f"Metadata for model '{model_name}' is None"
        shape = model_meta.inputs[0].shape
        channels, height, width = shape[1:]
        dtype = model_meta.inputs[0].datatype

        img_np = self.preprocess_image_pil(img)

        inputs = [
            grpcclient.InferInput(
                model_meta.inputs[0].name,
                [1, channels, height, width],
                dtype,
            )
        ]
        inputs[0].set_data_from_numpy(img_np.astype(triton_to_np_dtype(dtype)))
        outputs = [grpcclient.InferRequestedOutput(model_meta.outputs[0].name)]

        results = await self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        assert results is not None, "Inference returned None results"
        raw_output = results.as_numpy(model_meta.outputs[0].name)

        assert raw_output is not None, "Output tensor 'output' was not found in results"
        output = raw_output[0]

        cls_idx = np.argmax(output)
        cls_prob = output[cls_idx]

        inference = {
            "class_id": int(cls_idx),
            "probability": float(cls_prob),
        }

        return inference
