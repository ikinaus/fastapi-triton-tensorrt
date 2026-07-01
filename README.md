Go through the notebook file and create model.onnx and put the onnx file in the particular directory according to the Triton directory structure

#### Triton files architecture
```text
triton
└── models/
    └── classifier_onnx/
        ├── config.pbtxt    # configuration file
        └── 1/              # model version
            └── model.onnx  # model itself 
    └── classifier_trt/
        ├── config.pbtxt    # configuration file
        └── 1/              # model version
            └── model.plan  # model itself
```
#### Build and start the containers
```bash
docker compose up -d --build
```

#### To access the trtexec_container terminal:
```bash
docker exec -it trtexec_container bash
```

#### Command for ONNX -> TensorRT conversion (To be executed inside the trtexec_container):
```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:16x3x224x224 --fp16 --useSpinWait --outputIOFormats=fp16:chw --inputIOFormats=fp16:chw
```

Move new .plan file to the triton/models/classifier_trt/1 directory according triton file architecture

#### Interacting with the API Gateway
* **Interactive API:** Access the service endpoints manually at http://127.0.0.1:5000/docs ("Try it out")
* Ensure that you use the correct model name (`classifier_onnx` or `classifier_trt`).
