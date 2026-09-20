.PHONY: jaw_worm jaw_worm_mcts smoke dataset test build clean python-env

BUILD_DIR := build
PYTHON ?= python3
VENV := .venv/bin/python3
SEED ?= 1
SIMULATIONS ?= 2000

python-env:
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -r requirements.txt

smoke: build python-env
	PYTHONPATH=python $(VENV) -m sts_combat_rl.cli.smoke

dataset: build python-env
	PYTHONPATH=python $(VENV) -m sts_combat_rl.data.dataset data/mcts-slime-v2-pilot --seed-count 32 --simulations $(SIMULATIONS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

jaw_worm: build
	./$(BUILD_DIR)/play_jaw_worm $(SEED)

jaw_worm_mcts: build
	./$(BUILD_DIR)/play_jaw_worm_mcts $(SEED) $(SIMULATIONS)

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

clean:
	cmake -E remove_directory $(BUILD_DIR)
