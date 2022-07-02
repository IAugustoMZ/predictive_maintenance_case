FROM python:3.7.9
WORKDIR /root
RUN apt update && \
    apt install -y vim git curl
RUN pip install --upgrade pip
COPY ./docker/requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt
COPY ./modules /root/modules
COPY ./data /root/data
COPY ./analysis /root/analysis