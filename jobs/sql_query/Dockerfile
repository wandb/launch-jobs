# syntax=docker/dockerfile:1.4

# connectorx does not have linux ARM support
# https://github.com/sfu-db/connector-x/issues/240
FROM --platform=linux/amd64 python:3.9-buster
WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY --link job.py configs/ ./
ENTRYPOINT ["python", "job.py"]
