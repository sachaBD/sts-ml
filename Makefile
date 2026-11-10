.DEFAULT_GOAL := help
.PHONY: help env build test clean clean-data

BUILD_DIR := build/main

help: ## Show this help
	@echo "Usage: make <target>"
	@echo
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

env: ## Python environment (.venv) from pyproject.toml + uv.lock
	uv sync

build: ## Native build into build/main (all build dirs live under build/<name>/)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

test: build ## Build, then run every test (seconds)
	ctest --test-dir $(BUILD_DIR) --output-on-failure -j8 --timeout 60

clean: ## Remove build/main
	cmake -E remove_directory $(BUILD_DIR)

clean-data: RUN :=
clean-data: ## Preview Gumbel run cleanup: RUN=path; ARGS="--apply --drop-replay" to discard replay
	@test -n "$(RUN)" || { echo 'Usage: make clean-data RUN=path [ARGS="--apply --drop-replay"]'; exit 2; }
	@.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.tools.cleanup "$(RUN)" $(ARGS)

