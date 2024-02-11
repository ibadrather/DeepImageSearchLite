# Format code and check style
format:
	clear
	@echo "Formatting code using black"
	poetry run black .
	@echo "Checking code style with flake8"
	poetry run flake8 .

# Update pre-commit hooks
update-precommit:
	clear
	@echo "Updating pre-commit hooks"
	poetry run pre-commit autoupdate

# Run pre-commit hooks on all files
run-precommit:
	clear
	@echo "Running pre-commit hooks"
	poetry run pre-commit run --all-files

# Install project dependencies using poetry
install:
	clear
	@echo "Installing dependencies with Poetry"
	poetry install

# Update project dependencies to their latest versions
update-deps:
	clear
	@echo "Updating dependencies"
	poetry update

# Add a new dependency
# Usage: make add-dep package="somepackage>=1.2.3"
add-dep:
	clear
	@echo "Adding dependency $(package)"
	poetry add $(package)

# Add a new development dependency
# Usage: make add-dev-dep package="somepackage>=1.2.3"
add-dev-dep:
	clear
	@echo "Adding development dependency $(package)"
	poetry add $(package) --group dev

# Run tests
test:
	clear
	@echo "Running tests"
	poetry run pytest

# Clean up Python cache files and folders
clean:
	clear
	@echo "Cleaning up"
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +

# Initialize Poetry environment (useful for starting a new project)
init-poetry:
	clear
	@echo "Initializing new Poetry project"
	poetry init

# Show information about the current project
info:
	clear
	poetry show
