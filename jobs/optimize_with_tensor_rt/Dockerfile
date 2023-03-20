FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3
RUN pip install wandb
COPY optimize_with_tensorrt.py config.yml ./

ENTRYPOINT ["python3", "optimize_with_tensorrt.py"]