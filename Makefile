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


.PHONY: help clean clean-logs format sync pytest pytest-full train push coverage doctest
