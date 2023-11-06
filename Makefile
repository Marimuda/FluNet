# Include environment variables from .env file
include .env
export

# Default variables
PROJECT_NAME ?= flunet
VERSION ?= latest
LOCAL_DIR ?= $(PWD)

help:  ## Display this help message
	@echo "Available commands:"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean up autogenerated and temporary files
	@echo "Cleaning up auto-generated and temporary files..."
	@find htmlcov -type f ! -name '.gitignore' -delete
	@rm -rf dist
	@find . -type f -name "*.DS_Store" -ls -delete
	@find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	@find . | grep -E ".pytest_cache" | xargs rm -rf
	@find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	@rm -f .coverage
	@echo "Cleanup completed."

clean-logs: ## Remove log files
	@echo "Removing log files..."
	@rm -rf logs/**
	@echo "Log files removed."

format: ## Execute pre-commit hooks to enforce code style
	@echo "Running pre-commit hooks for code formatting..."
	@pre-commit run -a
	@echo "Code formatting completed."

sync: ## Synchronize with the main branch
	@echo "Syncing with the main branch..."
	@git pull origin main
	@echo "Sync complete."

pytest: ## Execute tests marked as 'not slow'
	@echo "Running fast test suite..."
	@coverage run --rcfile='tests/coverage.pytest.rc' -m pytest -k "not slow"
	@echo "Fast test suite completed."

pytest-full: ## Run the complete test suite
	@echo "Running full test suite..."
	@coverage run --rcfile='tests/coverage.pytest.rc' -m pytest
	@echo "Full test suite completed."

install: ## Install the project dependencies
	@echo "Installing project dependencies..."
	@pip install --upgrade pip && pip install -e .
	@echo "Dependencies installed."

setup-ci: ## Set up continuous integration environment
	@echo "Setting up continuous integration environment..."
	@pip install pre-commit && pre-commit install
	@echo "CI environment setup completed."

train: ## Start the model training process
	@echo "Training the model..."
	@python src/train.py
	@echo "Model training completed."

push: ## Execute pre-commit hooks, run full tests, and generate coverage reports
	@echo "Running pre-commit hooks, full tests, and generating coverage reports..."
	@make format pytest-full coverage
	@echo "Pre-push checks completed."

black: ## Format code with Black
	@echo "Formatting code with Black..."
	@pre-commit run black -a
	@echo "Code formatting with Black completed."

coverage:  ## Generate and display a coverage report
	@echo "Generating coverage report..."
	@coverage combine
	@coverage report --show-missing --omit=*test* --fail-under=80
	@coverage html
	@echo "Coverage report generated."

doctest: ## Run tests on docstrings
	@echo "Testing docstrings..."
	@coverage run --rcfile='tests/coverage.docstring.rc' -m pytest --doctest-modules src/flunet
	@echo "Docstring tests completed."

build_docker: ## Build the Docker image for the project
	@echo "Building Docker image '${PROJECT_NAME}:${VERSION}'..."
	@docker build --build-arg LOCAL_DIR=$(LOCAL_DIR) --build-arg TENSORBOARD_PORT=$(TENSORBOARD_PORT) --build-arg SERVICE_PORT=$(SERVICE_PORT) -t $(PROJECT_NAME):$(VERSION) .
	@echo "Docker image build completed."

run_docker: ## Run the Docker container for the project
	@echo "Starting Docker container from image '${PROJECT_NAME}:${VERSION}'..."
	@docker run --workdir $(DOCKER_WORK_DIR) $(GPU_FLAG) --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v $(LOCAL_DIR):$(DOCKER_WORK_DIR) -p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT) -p $(SERVICE_PORT):$(SERVICE_PORT) $(PROJECT_NAME):$(VERSION)
	@echo "Docker container is now running."

stop_docker: ## Stop the running Docker container
	@echo "Stopping Docker container for '${PROJECT_NAME}'..."
	@docker stop $(PROJECT_NAME)
	@echo "Docker container stopped."

clean_docker: ## Remove the specified Docker image
	@echo "Removing Docker image '${PROJECT_NAME}:${VERSION}'..."
	@docker rmi $(PROJECT_NAME):$(VERSION)
	@echo "Docker image removed."

sanity_test_docker: ## Execute sanity tests within the Docker container
	@echo "Running sanity tests inside Docker container..."
	@docker exec -it $(PROJECT_NAME) pytest $(DOCKER_WORK_DIR)/tests/datasets
	@echo "Sanity tests completed."

.PHONY: help clean clean-logs format sync pytest pytest-full install setup-ci train push black coverage doctest build_docker run_docker stop_docker clean_docker sanity_test_docker
