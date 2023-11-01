# Include environment variables from .env file
include .env
export

# Default variables
PROJECT_NAME ?= flunet
VERSION ?= latest
LOCAL_DIR ?= $(PWD)


help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	find htmlcov -type f ! -name '.gitignore' -delete
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

clean-logs: ## Clean logs
	rm -rf logs/**

format: ## Run pre-commit hooks
	pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main

pytest: ## Run not slow tests
	coverage run \
                --rcfile='tests/coverage.pytest.rc' \
                -m pytest -k "not slow"

pytest-full: ## Run all tests
	coverage run \
                --rcfile='tests/coverage.pytest.rc' \
                -m pytest
train: ## Train the model
	python src/train.py

push: ## Run pre-commit hooks, full tests, and generate coverage
	format test-full coverage

coverage:  ## Generate coverage report
	coverage combine && \
		coverage report --show-missing --omit=*test* --fail-under=80 && \
		coverage html

doctest:
	coverage run \
                --rcfile='tests/coverage.docstring.rc' \
                -m pytest \
                --doctest-modules src/flunet


# Check for essential environment variables
ifeq ($(strip $(PROJECT_NAME)),)
$(error PROJECT_NAME is not set. Check .env file.)
endif

build: ## Build the Docker image
	@echo "Building Docker image..."
	docker build \
		--build-arg LOCAL_DIR=$(LOCAL_DIR) \
		--build-arg TENSORBOARD_PORT=$(TENSORBOARD_PORT) \
		--build-arg SERVICE_PORT=$(SERVICE_PORT) \
		-t $(PROJECT_NAME):$(VERSION) .

run: ## Run the Docker container
	@echo "Running Docker container..."
	@echo "LOCAL_DIR is $(LOCAL_DIR) and DOCKER_WORK_DIR is $(DOCKER_WORK_DIR)"
	
	docker run --workdir $(DOCKER_WORK_DIR) $(GPU_FLAG) --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it \
		-v $(LOCAL_DIR):$(DOCKER_WORK_DIR) \
		-p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT) \
		-p $(SERVICE_PORT):$(SERVICE_PORT) \
		$(PROJECT_NAME):$(VERSION)

stop: ## Stop the Docker container
	@echo "Stopping Docker container..."
	docker stop $(PROJECT_NAME)

clean: ## Remove the Docker image
	@echo "Removing Docker image..."
	docker rmi $(PROJECT_NAME):$(VERSION)

docker-sanity-test: ## Run tests inside the Docker container
	@echo "Running sanity tests..."
	docker exec -it $(PROJECT_NAME) pytest $(DOCKER_WORK_DIR)/tests/datasets


.PHONY: help clean clean-logs format sync pytest pytest-full train push coverage doctest build run stop clean docker-sanity-test