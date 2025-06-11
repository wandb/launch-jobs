FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-distutils \
        build-essential git curl ca-certificates cmake \
        python3-setuptools

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv ~/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml .

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -r pyproject.toml

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "evals.py"]