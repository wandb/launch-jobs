# syntax=docker/dockerfile:1.4

FROM python:3.9-slim
RUN apt update && apt install gcc -y

WORKDIR /setup
COPY --link requirements.txt ./
RUN apt update
RUN apt install gcc git git-lfs -y
RUN git lfs install
RUN pip install -r requirements.txt
RUN git clone https://github.com/openai/evals.git ./evals
# extra metrics
RUN pip install -U -e ./evals &&
    cd evals && git checkout 456721983f6c116b27242cadd62f57635f1bc76e &&
    git lfs fetch --all && git lfs pull
COPY --link openai.py ./evals/evals/completion_fns/openai.py

WORKDIR /launch
COPY --link job.py configs/ ./

ENTRYPOINT ["python3", "job.py"]
