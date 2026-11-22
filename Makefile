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


# Browser-only combat inspector. Activate Emscripten before building.
LIGHTSPEED ?= $(abspath ../lightspeed)
VIEWER_SITE ?= build/gumbel-site
VIEWER_RUNTIME ?= build/web-deps/node_modules/onnxruntime-web
.PHONY: pages-wasm pages-runtime pages-build pages-preview pages-publish
pages-wasm: ## Compile current simulator, encodings and inspector to browser WASM
	emcmake cmake -S apps/combat_inspector/wasm -B build/gumbel-wasm -DCMAKE_BUILD_TYPE=Release -DSTS_LIGHTSPEED_DIR="$(LIGHTSPEED)"
	cmake --build build/gumbel-wasm --parallel 4

pages-runtime: ## Download the pinned browser inference runtime under build/
	npm install --prefix build/web-deps --no-audit --no-fund --package-lock=false onnxruntime-web@1.23.2

pages-build: pages-wasm ## Freeze CHECKPOINT=... and FIGHTS=... for GitHub Pages (no publishing)
	@test -n "$(CHECKPOINT)" -a -n "$(FIGHTS)" || { echo 'Set CHECKPOINT and FIGHTS'; exit 2; }
	.venv/bin/python -m apps.combat_inspector.pages build --checkpoint "$(CHECKPOINT)" --fights "$(FIGHTS)" --simulator "$(LIGHTSPEED)" --runtime "$(VIEWER_RUNTIME)" --site "$(VIEWER_SITE)"

pages-preview: ## Serve the prepared viewer at http://localhost:8734/gumbel-tree/
	.venv/bin/python -m http.server 8734 --bind 127.0.0.1 --directory "$(VIEWER_SITE)"

pages-publish: ## Explicitly publish the prepared viewer to gh-pages; preserves other reports
	.venv/bin/python -m apps.combat_inspector.pages publish --site "$(VIEWER_SITE)"
