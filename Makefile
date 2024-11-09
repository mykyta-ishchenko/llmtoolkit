.PHONY: lint fix python-version

lint:
	@echo "Running linters..."
	poetry run ruff format .
	poetry run ruff check .

fix:
	@echo "Running linters with auto-fix..."
	poetry run ruff format .
	poetry run ruff check --fix .

python-version:
	@grep 'python = ' pyproject.toml | awk -F'"' '{print $$2}' | sed 's/[^0-9.]//g'
