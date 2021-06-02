FROM pytorch/pytorch:latest

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt