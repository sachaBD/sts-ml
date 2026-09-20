.PHONY: jaw_worm jaw_worm_mcts smoke dataset build clean python-env

BUILD_DIR := build
PYTHON ?= python3.13
VENV := .venv/bin/python3
SEED ?= 1
SIMULATIONS ?= 2000

python-env:
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -r requirements.txt

smoke: build python-env
	$(VENV) python/smoke.py ./$(BUILD_DIR)/dump_slime_boss_encoding

dataset: build python-env
	$(VENV) python/generate_mcts_dataset.py data/mcts-slime-v1-pilot --seed-count 32 --simulations $(SIMULATIONS)

jaw_worm: build
	./$(BUILD_DIR)/play_jaw_worm $(SEED)

jaw_worm_mcts: build
	./$(BUILD_DIR)/play_jaw_worm_mcts $(SEED) $(SIMULATIONS)

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

clean:
	cmake -E remove_directory $(BUILD_DIR)
