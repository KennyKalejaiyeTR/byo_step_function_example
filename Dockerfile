FROM python:3.7.9-slim-buster

COPY requirements.txt ./

RUN apt-get update \
    && apt-get install -y gcc \
    && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]