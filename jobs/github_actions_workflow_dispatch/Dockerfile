# syntax=docker/dockerfile:1.4

FROM python:3.9-slim
WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY --link job.py configs/ ./
ENTRYPOINT ["python", "job.py"]
