FROM python:3.9-buster
WORKDIR /launch

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY job.py config.yml ./
ENTRYPOINT ["python", "job.py"]