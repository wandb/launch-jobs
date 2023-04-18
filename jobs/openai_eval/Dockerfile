FROM python:3.9-buster

WORKDIR /setup
COPY requirements.txt ./
RUN apt update && \
    apt install git-lfs && \
    git lfs install
RUN pip install -r requirements.txt
RUN git clone https://github.com/openai/evals.git ./evals && \
    # extra metrics
    COPY openai.py ./evals/evals/completion_fns/openai.py
RUN pip install -U -e ./evals && \
    cd evals && git lfs fetch --all && git lfs pull

WORKDIR /launch
COPY job.py config.yml ./

ENTRYPOINT ["python3", "job.py"]