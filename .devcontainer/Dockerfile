FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV WORKDIR=/workspaces/mlops-cifar-classifier
ENV MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
ENV MLFLOW_ARTIFACT_ROOT=/workspaces/mlops-cifar-classifier/mlflow-artifacts


USER root

RUN apt-get update  --allow-insecure-repositories\
    && apt-get install -y --no-install-recommends \
        git-all \
        curl \
        wget \
        htop \
        vim \
        build-essential \
        python3-pip \
        tmux \
    && apt-get upgrade -y \
    && apt-get autoremove -y\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip poetry==1.8.3

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false
RUN poetry install
