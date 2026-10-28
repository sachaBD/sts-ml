Multi-deck combat expert iteration using agents/combat/nn. One shared transformer,
Gumbel search, and fixed monitoring against approximate historical teacher scores.

out/config.toml and command.json record configuration and the NN launch command.
out/inputs contains named combat_v4 starts, monitoring starts/reference outcomes,
and a copied nn_play executable. out/training contains the NN loop's checkpoints,
binary samples, ONNX models/engines, results and logs. The app dashboard reads these
files directly. No Parquet tables or combat_v4 action traces are generated.

Earlier runs may also contain prepared.json, summary.json and REPORT.md from the
original wrapper. These historical artifacts remain unchanged.
