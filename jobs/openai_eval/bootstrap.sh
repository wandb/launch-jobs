git clone https://github.com/openai/evals.git ./evals
pip install -U -e ./evals
cd evals && \
    git lfs fetch --all && \
    git lfs pull
