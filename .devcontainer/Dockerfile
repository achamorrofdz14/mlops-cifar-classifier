FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV WORKDIR=/workspaces/mlops-cifar-classifier

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        htop \
        vim \
        build-essential \
        python3-pip \
        libcudnn8 \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip poetry==1.8.3

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install
