# Makefile for ML Service v2

.PHONY: help install test lint format clean build proto

help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linters"
	@echo "  format    - Format code"
	@echo "  proto     - Generate proto files"
	@echo "  clean     - Clean temporary files"
	@echo "  build     - Build Docker image"

install:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	ruff check src/
	mypy src/

format:
	black src/ tests/
	ruff format src/ tests/

proto:
	./generate_proto.sh

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

build:
	docker build -t video-translation-ml-v2 .



