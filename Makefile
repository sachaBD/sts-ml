.DEFAULT_GOAL := help
.PHONY: help env env-onnx build test clean clean-data

BUILD_DIR := build/main

help: ## Show this help
	@echo "Usage: make <target>"
	@echo
	@echo "Python environment (.venv, from pyproject.toml + uv.lock via uv)."
	@echo "Each env target makes .venv contain exactly that set; anything else is uninstalled."
	@echo
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

env: ## Core only: torch (CUDA 12.6), numpy, pyarrow
	uv sync

env-onnx: ## Core + ONNX export/inference (needed by the app)
	uv sync --group onnx

build: ## Native build into build/main (all build dirs live under build/<name>/)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

test: build ## Build, then run native, Python and dashboard UI tests through CTest
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean: ## Remove build/main
	cmake -E remove_directory $(BUILD_DIR)

clean-data: RUN :=
clean-data: ## Preview Gumbel run cleanup: RUN=path; ARGS="--apply --drop-replay" to discard replay
	@test -n "$(RUN)" || { echo 'Usage: make clean-data RUN=path [ARGS="--apply --drop-replay"]'; exit 2; }
	@.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.cleanup "$(RUN)" $(ARGS)

