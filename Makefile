.PHONY: build test clean python-env jax-env

BUILD_DIR := build/main
PYTHON ?= python3
VENV := .venv/bin/python3

python-env:
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -r requirements.txt

# JAX stack for agents/combat/az; jaxsim from the sts-lightspeed checkout (same override as CMake).
STS_LIGHTSPEED_DIR ?= ../sts_lightspeed
jax-env:
	STS_LIGHTSPEED_DIR=$(abspath $(STS_LIGHTSPEED_DIR)) $(VENV) -m pip install -r requirements-jax.txt

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake -E remove_directory $(BUILD_DIR)
