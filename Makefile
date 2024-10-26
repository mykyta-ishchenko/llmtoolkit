.PHONY: lint fix

lint:
	@echo "Running linters..."
	poetry run ruff format .
	poetry run ruff check .

fix:
	@echo "Running linters with auto-fix..."
	poetry run ruff format .
	poetry run ruff check --fix .
