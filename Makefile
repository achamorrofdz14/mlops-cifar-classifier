
PYTHON=python3

.PHONY: tests
tests:
	pytest -v -s --disable-warnings --cov=./src/cifar_classifier/data --cov-report=xml:./reports/coverage.xml tests

.PHONY: mlflow-start
mlflow-start:
	mlflow server

.PHONY: data
data:
	${PYTHON} src/cifar_classifier/data/process_data.py --train --validation

.PHONY: train
train: data
	${PYTHON} src/cifar_classifier/model/train_model.py --experiment-name dev_cifar_classifier --run-reason development --team "Data Scientists"

.PHONY: conf-env
conf-env:
	poetry lock
	poetry install
