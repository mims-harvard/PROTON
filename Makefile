.DEFAULT_GOAL := help

.PHONY: help
help: ##@ List available commands with their descriptions
	@printf "\nUsage: make <command>\n"
	@grep -F -h "##@" $(MAKEFILE_LIST) | grep -F -v grep -F | sed -e 's/\\$$//' | awk 'BEGIN {FS = ":*[[:space:]]*##@[[:space:]]*"}; \
	{ \
		if($$2 == "") \
			pass; \
		else if($$0 ~ /^#/) \
			printf "%s", $$2; \
		else if($$1 == "") \
			printf "     %-20s%s", "", $$2; \
		else \
			printf "    \033[34m%-20s\033[0m %s\n", $$1, $$2; \
	}'

.PHONY: install
install: ##@ Create the virtual environment and install the pre-commit hooks
	@echo "Creating virtual environment using uv..."
	@uv sync
	@echo "Checking for CUDA availability..."
	@uv run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && \
		(echo "CUDA is available. Installing DGL with CUDA support..." && \
		 uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html) || \
		(echo "CUDA is not available. Syncing with CPU extras..." && \
		 if [ "$$(uname -s)" = "Linux" ]; then \
		 	echo "Linux detected. Installing DGL via pip..." && \
		 	uv pip install "dgl==2.2.1" -f https://data.dgl.ai/wheels/torch-2.3/repo.html; \
		 else \
		 	uv run install-dgl || (echo "DGL installation skipped or failed. Run 'make install-dgl' manually if needed." && true); \
		 fi)
	@uv run pre-commit install

.PHONY: install-dgl
install-dgl: ##@ Install DGL from source (for macOS ARM64 or when wheels are unavailable)
	@echo "Installing DGL from source..."
	@uv run install-dgl

.PHONY: check
check: ##@ Run code quality tools.
	@echo "Checking lock file consistency with 'pyproject.toml'..."
	@uv_lock_was_clean=$$(git diff --quiet uv.lock 2>/dev/null && echo "1" || echo "0"); \
	if ! uv lock --locked 2>&1; then \
		echo "ERROR: uv.lock is out of date. Please run 'uv lock' to update it."; \
		exit 1; \
	fi; \
	if [ "$$uv_lock_was_clean" = "1" ] && ! git diff --quiet uv.lock 2>/dev/null; then \
		echo "WARNING: uv.lock was modified by 'uv lock --locked'. Restoring it."; \
		git checkout -- uv.lock; \
	fi
	@echo "Linting code..."
	@uv run pre-commit run -a || true
	@modified_files=$$(git status --porcelain | grep -v '^ M uv.lock' || true); \
	if [ -n "$$modified_files" ]; then \
		echo "ERROR: Pre-commit hooks modified files. This means files are not properly formatted."; \
		echo "Please run 'pre-commit run -a' locally and commit the changes."; \
		echo "Modified files:"; \
		echo "$$modified_files"; \
		exit 1; \
	fi
	@echo "Static type checking..."
	@uv run ty check --exclude "notebooks/**"

.PHONY: tox
tox: ##@ Run tox to test the code with all supported Python versions
	@uv run tox

.PHONY: build
build: ##@ Build wheel file
	@make clean-build
	@echo "Creating wheel file..."
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ##@ Clean build artifacts
	@echo "Removing build artifacts..."
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: clean
clean: ##@ Clean up the project
	@echo "Cleaning up the project..."
	@rm -rf `find . -name __pycache__`
	@rm -f `find . -type f -name '*.py[co]'`
	@rm -f `find . -type f -name '*~'`
	@rm -f `find . -type f -name '.*~'`
	@rm -f `find . -type f -name '*.log'`
	@rm -rf `find . -type d -name '.ipynb_checkpoints'`
	@rm -rf `find . -type d -name '*.egg-info'`
	@rm -rf .cache
	@rm -rf dist
	@rm -rf .mypy_cache
	@rm -rf .venv
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf htmlcov
	@rm -f .coverage*
	@rm -rf target
	@rm -rf .tox

.PHONY: jupyterlab
jupyterlab: ##@ Spin up JupyterLab
	@uv run jupyter lab

.PHONY: run-rmd
run-rmd: ##@ Render an R Markdown (.Rmd) file using Rscript. Example usage: RMD_FILE="path/to/file.Rmd" make run-rmd
	@uv run Rscript -e "rmarkdown::render('${RMD_FILE}')"
