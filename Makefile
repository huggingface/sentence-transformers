.PHONY: venv
venv: ## Create virtual environment and install development dependencies
	@echo "Creating virtual environment..."
	@python -m venv .venv
	@echo "Activating virtual environment..."
ifeq ($(OS), Windows_NT)
	.venv\Scripts\activate
else
	. .venv/bin/activate
endif
	@echo "Installing development dependencies..."
	@pip install '.[dev]'

.PHONY: check
check: ## Run code quality tools
	@echo "Linting code via pre-commit"
	@pre-commit run -a

.PHONY: test
test: ## Run unit tests
	@pytest

.PHONY: test-cov
test-cov: ## Run unit tests and generate a coverage report
	@pytest --cov-report term --cov-report=html --cov=sentence_transformers

.PHONY: build
build: ## Build the wheel file
	@echo "Building the wheel file..."
	@pip install build
	@python -m build

.PHONY: help
help: ## Show help for the commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
