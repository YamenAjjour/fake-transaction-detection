FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#RUN apt-get update && \
#apt-get install -y python3 python3-pip
RUN mkdir -p /fake-transaction-detector
COPY . /fake-transaction-detector
WORKDIR /fake-transaction-detector

RUN pip install torch transformers datasets pandas scikit-learn fastapi



CMD ["fastapi", "run", "/fake-transaction-detector/server.py", "--port", "8000"]


EXPOSE 8000




