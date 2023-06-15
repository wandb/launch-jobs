FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV RANK 0
ENV LOCAL_RANK 0
ENV WORLD_SIZE 1
ENV MASTER_ADDR localhost
ENV MASTER_PORT 12345
RUN pip install wandb
ADD train.py /src/train.py
ENTRYPOINT ["python", "/src/train.py"]