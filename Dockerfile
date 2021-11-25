FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ENV USERNAME=root

COPY requirements.txt requirements.txt
RUN pip install -U pip && pip install -r requirements.txt
