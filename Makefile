.DEFAULT_GOAL := help
.PHONY: help env env-onnx env-jax env-all build test clean clean-data infra-benchmark

BUILD_DIR := build/main

help: ## Show this help
	@echo "Usage: make <target>"
	@echo
	@echo "Python environment (.venv, from pyproject.toml + uv.lock via uv)."
	@echo "Each env target makes .venv contain exactly that set; anything else is uninstalled."
	@echo
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

env: ## Core only: torch (CUDA 12.6), numpy, pyarrow, msgpack, scipy, duckdb
	uv sync

env-onnx: ## Core + ONNX export/inference (agents/combat/pv)
	uv sync --group onnx

env-jax: ## Core + JAX GPU self-play stack + jaxsim from ../lightspeed (agents/combat/az)
	uv sync --group jax

env-all: ## Core + ONNX + JAX: everything
	uv sync --all-groups

build: ## Native build into build/main (all build dirs live under build/<name>/)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

test: build ## Build, then run native + Python tests through CTest
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# infra-benchmark: inference always on the fixed panache 128x4 model (see its README); the Reader reads the
# replay window of the newest training run, or RUN=.../out/training.
BENCH_MODEL ?= runs/benchmark/panache-128x4
RUN ?= $(lastword $(sort $(wildcard runs/schema=gumbel_nn_combat_v1/date=*/id=*/out/training)))

infra-benchmark: ## Component benchmarks for infra/ (about 20 s)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release > /dev/null
	@cmake --build $(BUILD_DIR) --target infra_bench_inference infra_bench_actors --parallel > /dev/null
	@echo "model:  $(BENCH_MODEL)"
	@echo "replay: $(RUN)"
	@.venv/bin/python -m infra.benchmarks "$(BENCH_MODEL)" $(wildcard $(RUN)/iter*/samples-0.bin) --build $(BUILD_DIR)

clean: ## Remove build/main
	cmake -E remove_directory $(BUILD_DIR)

# Do not inherit infra-benchmark's default RUN: cleanup requires an explicit target.
clean-data: RUN :=
clean-data: ## Preview Gumbel run cleanup: RUN=path; ARGS="--apply --drop-replay" to discard replay
	@test -n "$(RUN)" || { echo 'Usage: make clean-data RUN=path [ARGS="--apply --drop-replay"]'; exit 2; }
	@.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.cleanup "$(RUN)" $(ARGS)
