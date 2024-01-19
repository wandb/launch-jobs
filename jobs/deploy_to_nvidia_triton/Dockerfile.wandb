# syntax=docker/dockerfile:1.4

FROM python:3.9-slim
RUN apt update && apt install gcc -y

RUN apt-get update && apt-get install -y --no-install-recommends build-essential
# RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
# RUN apt-get -y install curl
# RUN apt-get install libgomp1
WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY --link job.py configs/ ./
ENTRYPOINT ["python3", "job.py"]
