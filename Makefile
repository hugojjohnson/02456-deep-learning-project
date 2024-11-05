#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = 02456-deep-learning-project
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
<<<<<<< HEAD
	flake8 02456_deep_learning_project
	isort --check --diff --profile black 02456_deep_learning_project
	black --check --config pyproject.toml 02456_deep_learning_project
=======
	flake8 src
	isort --check --diff --profile black src
	black --check --config pyproject.toml src
>>>>>>> 4cc3298 (Set up with ccds)

## Format source code with black
.PHONY: format
format:
<<<<<<< HEAD
	black --config pyproject.toml 02456_deep_learning_project
=======
	black --config pyproject.toml src
>>>>>>> 4cc3298 (Set up with ccds)




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


<<<<<<< HEAD
=======
## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py

>>>>>>> 4cc3298 (Set up with ccds)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
