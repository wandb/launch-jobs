# syntax=docker/dockerfile:1.4

FROM python:3.9-slim
WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY --link deploy_to_sagemaker_endpoints.py inference.py ./
COPY --link config_pytorch.yml config_tensorflow.yml ./
ENTRYPOINT ["python", "deploy_to_sagemaker_endpoints.py"]