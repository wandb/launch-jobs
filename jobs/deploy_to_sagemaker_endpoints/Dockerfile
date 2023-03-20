FROM python:3.9-buster
WORKDIR /launch

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY deploy_to_sagemaker_endpoints.py inference.py ./
COPY config_pytorch.yml config_tensorflow.yml ./
ENTRYPOINT ["python", "deploy_to_sagemaker_endpoints.py"]