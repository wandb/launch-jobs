FROM python:3.9-buster
# RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
# RUN apt-get -y install curl
# RUN apt-get install libgomp1
WORKDIR /launch

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY job.py config_pytorch.yml config_tensorflow.yml ./
ENTRYPOINT ["python3", "job.py"]