
format:
	clear
	@echo "Formatting code using black"
	python3 -m poetry run black .
	python3 -m poetry run flake8 .
