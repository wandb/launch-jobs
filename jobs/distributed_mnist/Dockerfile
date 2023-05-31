FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ENV RANK 0
ENV LOCAL_RANK 0
ENV WORLD_SIZE 1
ENV MASTER_ADDR localhost
ENV MASTER_PORT 1234
ENV PL_TORCH_DISTRIBUTED_BACKEND gloo
RUN pip install wandb
ADD train.py /src/train.py
ENTRYPOINT ["python", "/src/train.py"]