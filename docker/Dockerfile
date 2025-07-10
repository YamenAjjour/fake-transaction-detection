from ubuntu:latest

RUN sudo apt-get --upgrade
RUN sudo apt-get install-y python-pip3
RUN pip install --r requirements.txt

RUN mkdir -p /fake-transaction-detector

COPY . /fake-transaction-detector

RUN fastapi dev /fake-transaction-detector/server.py





