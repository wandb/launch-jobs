# syntax=docker/dockerfile:1.4

FROM python:3.9-slim
RUN apt update && apt install gcc -y

WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY --link data_loader.py ./
RUN python data_loader.py  # loads the keras data

COPY --link job.py configs/ ./
ENTRYPOINT ["python", "job.py"]
