.PHONY: tests

tests:
	pytest -v tests

conf-env:
	poetry lock
	poetry install