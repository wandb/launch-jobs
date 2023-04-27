# syntax=docker/dockerfile:1.4

FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3
RUN pip install -r requirements.txt
COPY --link job.py configs/ ./

ENTRYPOINT ["python3", "job.py"]
