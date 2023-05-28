FROM nvcr.io/nvidia/tritonserver:22.11-py3
RUN pip install torchvision opencv-python-headless
ENTRYPOINT ["tritonserver", "--model-repository", "s3://andrew-triton-bucket/models", "--model-control-mode", "explicit"]
