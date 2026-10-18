.PHONY: help install test run-api run-ui evaluate localize docker-build

PYTHON ?= python3

help:
	@echo "VulnLocator - AI-Powered Vulnerability Detection & Localization"
	@echo "================================================================"
	@echo "Available commands:"
	@echo "  make install       Install dependencies in active Python environment"
	@echo "  make test          Run unit and integration test suite"
	@echo "  make run-api       Start the FastAPI backend server (port 8000)"
	@echo "  make run-ui        Start the Streamlit user interface (port 8501)"
	@echo "  make evaluate      Evaluate model against held-out PrimeVul test split"
	@echo "  make localize      Run CLI localization demo with sample SQL injection"
	@echo "  make docker-build  Build the production Docker image"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover tests -v

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run app/ui.py

evaluate:
	$(PYTHON) -m src.evaluate --csv data/primevul_sample.csv

localize:
	$(PYTHON) -m src.localize
