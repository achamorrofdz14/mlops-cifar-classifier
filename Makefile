
PYTHON=python3

.PHONY: tests
tests:
	pytest -v -s --disable-warnings --cov=./src/cifar_classifier/data --cov-report=xml:./reports/coverage.xml tests

.PHONY: mlflow
mlflow:
	mlflow server

.PHONY: data
data:
	@if [ ! -f "data/train_dataset.pickle" ] || [ ! -f "data/val_dataset.pickle" ]; then \
		${PYTHON} src/cifar_classifier/data/process_data.py --train --validation; \
	else \
		echo "Processed data already exists. Skipping data processing."; \
	fi

.PHONY: train
train: data
	${PYTHON} src/cifar_classifier/model/train_model.py --experiment-name dev_cifar_classifier --run-reason development --team "Data Scientists"

.PHONY: conf-env
conf-env:
	poetry lock
	poetry install
