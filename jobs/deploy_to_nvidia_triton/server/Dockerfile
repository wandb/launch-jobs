# syntax=docker/dockerfile:1.4

FROM nvcr.io/nvidia/tritonserver:22.11-py3
RUN apt update && apt install libgl1 -y

COPY --link requirements.txt ./
RUN pip install -r ./requirements.txt

ENTRYPOINT ["tritonserver", "--model-repository", "s3://andrew-triton-bucket/models", "--model-control-mode", "explicit"]
