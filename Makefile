.PHONY: build test clean python-env

BUILD_DIR := build/main
PYTHON ?= python3
VENV := .venv/bin/python3

python-env:
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -r requirements.txt

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure
	PYTHONPATH=python:. $(VENV) -m unittest discover -s tests

clean:
	cmake -E remove_directory $(BUILD_DIR)
