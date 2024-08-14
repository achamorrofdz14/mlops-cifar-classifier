FROM python:3.11-bullseye

ARG DEBIAN_FRONTEND=noninteractive

ENV WORKDIR=/workspace/mlops-cifar-classifier

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        htop \
        vim \
        build-essential \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip poetry==1.8.3

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root 

