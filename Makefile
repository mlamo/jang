install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests

test-cov:
	pytest --cov src/jang --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage tests

.PHONY: black
black:
	black src/jang

.PHONY: black-check
black-check:
	black --check src/jang

.PHONY: all clean install install-dev black black-check
