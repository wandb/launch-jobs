FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3
RUN pip install wandb
COPY job.py config.yml ./

ENTRYPOINT ["python3", "job.py"]