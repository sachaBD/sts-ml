# Shared predictive models

Models here predict outcomes rather than choose actions, and have concrete consumers outside any one agent.

`combat_outcome/` predicts pre-combat win probability and final-HP distributions. It is shared by overworld rollout search, evaluation tools, and the GUI. Its owner includes architecture, feature preparation, learning, and inference helpers. Agent-specific models live with their agent instead.

Saved checkpoints belong in `runs/`.
