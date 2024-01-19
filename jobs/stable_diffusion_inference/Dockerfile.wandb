FROM huggingface/transformers-pytorch-gpu:4.35.0
RUN pip install diffusers accelerate wandb
COPY inference.py /
ENTRYPOINT [ "python3", "/inference.py"]
