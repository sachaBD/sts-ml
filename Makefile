.PHONY: build test clean python-env

BUILD_DIR := build/main
VENV := .venv/bin/python3

# Python env from pyproject.toml/uv.lock. Optional groups: make python-env GROUPS="--group jax" (or --all-groups).
GROUPS ?=
python-env:
	uv sync $(GROUPS)

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake -E remove_directory $(BUILD_DIR)
