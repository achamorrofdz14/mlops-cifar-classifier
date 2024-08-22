.PHONY: tests
tests:
	pytest -v --disable-warnings tests

.PHONY: conf-env
conf-env:
	poetry lock
	poetry install