from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from infer_triton import InferenceModule

app = FastAPI()

with open("imagenet_classes.txt", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

inference_module = InferenceModule()


@app.post("/predict", description="Initializing image classification...")
async def predict(
    file: UploadFile = File(..., description="Image for classification."),
    model_name: str = Query(..., description="Triton's model name"),
):
    try:
        # Reading img as byte
        contents = await file.read()

        # Triton inference call
        result = await inference_module.infer_image(contents, model_name=model_name)

        class_id = result["class_id"]
        probability = result["probability"]

        class_name = class_names[class_id]

        return {
            "class_id": class_name,
            "class_name": class_name,
            "probability": probability,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error:{str(e)}")
