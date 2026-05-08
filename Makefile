
.PHONY: check
check: ## Run code quality tools.
	@echo "Linting code via pre-commit"
	@pre-commit run -a

.PHONY: test
test: ## Run unit tests
	@pytest

.PHONY: test-cov
test-cov: ## Run unit tests and generate a coverage report
	@pytest --cov-report term --cov-report=html --cov=sentence_transformers

.PHONY: docs
docs: ## Build the documentation
	@$(MAKE) -C docs docs

.PHONY: help
help: ## Show help for the commands.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
