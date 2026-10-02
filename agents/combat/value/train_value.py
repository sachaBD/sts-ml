"""Value trainer (apps/value_train). Deterministic with threads = 1 on the CPU; optionally on a CUDA GPU."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agents.combat.value import ENCODING_VERSIONS, DeepSetsV2, DeepSetsV3, build

from runs.run import run_dir
from environments.combat import schema as combat_v3
from . import data_v4
from .data import RowLoader, Rows, assign_aux_targets, assign_targets, split_rows, training_rows


LABELS = ("blend", "shift", "root", "terminal")  # see data.assign_targets


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Every training setting, all explicit. The None-able ones are features that are off when None, and
    are required exactly when they apply (checked in __post_init__)."""
    data: str                        # SQL over runs.query selecting combat_v3 rows
    oracle: bool                     # allow oracle (perfect-foresight) rows
    model: dict[str, Any]            # architecture dict for agents.combat.value.build: kind + every argument
    output: Path                     # checkpoint path; its .json and training_history.json go next to it
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    label: str                       # LABELS
    blend: float | None              # weight on terminal_value; label blend / shift only
    lr_schedule: str                 # constant, or cosine (per-step, lr -> 0)
    keep: str                        # last: checkpoint of the last epoch; best: of the best validation MSE
    threads: int                     # torch intra-op CPU threads (> 1 may be slightly nondeterministic)
    device: str                      # cpu or cuda[:N]
    aux_won_weight: float            # > 0 needs a model with aux heads
    aux_hp_weight: float
    initial_checkpoint: Path | None  # start from these weights (fresh optimizer)
    split: str | None                # with initial_checkpoint: pinned (its train/validation runs) or fresh
    validation_fraction: float | None  # of run seeds; unless split = pinned
    corrections: str | None          # SQL selecting DAgger rows (target root_value); needs split = pinned
    correction_weight: float | None  # with corrections: share of the loss on correction rows
    row_weighting: str | None        # optional encounter-balanced sqrt fight-length weighting
    aux_keep_weight: float | None    # deep_sets_v3 only: potions-kept head (won rows holding a potion)
    policy_weight: float | None      # deep_sets_v3 only: policy cross entropy (0 with policy_width 0)

    def __post_init__(self) -> None:
        types = {"str": str, "bool": bool, "int": int, "float": (int, float), "Path": Path, "dict[str, Any]": dict}
        for field in dataclasses.fields(self):
            value, kind = getattr(self, field.name), field.type.removesuffix(" | None")
            if value is None and field.type.endswith(" | None"):
                continue
            if not isinstance(value, types[kind]) or (isinstance(value, bool) and kind != "bool"):
                raise TypeError(f"{field.name} must be {kind}, got {value!r}")
        for name, allowed in (("label", LABELS), ("lr_schedule", ("constant", "cosine")), ("keep", ("last", "best")),
                              ("split", (None, "pinned", "fresh", "stable"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} {getattr(self, name)!r} is not one of {allowed}")
        if self.row_weighting not in (None, "elite_sqrt_source"):
            raise ValueError(f"unknown row_weighting {self.row_weighting!r}")
        if "kind" not in self.model:
            raise ValueError("model needs a kind")

        def exactly_when(name: str, applies: bool, when: str) -> None:
            if applies != (getattr(self, name) is not None):
                raise ValueError(f"{name} is required {when}, and only then")

        exactly_when("blend", self.label in ("blend", "shift"), "with label blend / shift")
        exactly_when("split", self.initial_checkpoint is not None, "with initial_checkpoint")
        exactly_when("validation_fraction", self.split != "pinned", "unless split = pinned")
        exactly_when("correction_weight", self.corrections is not None, "with corrections")
        v3 = self.model["kind"] == "deep_sets_v3"
        exactly_when("aux_keep_weight", v3, "with model kind deep_sets_v3")
        exactly_when("policy_weight", v3, "with model kind deep_sets_v3")
        if v3 and self.policy_weight > 0 and not self.model.get("policy_width"):
            raise ValueError("policy_weight > 0 needs a policy head (policy_width > 0)")
        if self.corrections is not None and self.split != "pinned":
            raise ValueError("corrections need an initial checkpoint with split = 'pinned'")
        if self.correction_weight is not None and not 0 <= self.correction_weight <= 1:
            raise ValueError(f"correction_weight {self.correction_weight} outside [0, 1]")


def atomic_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def resolve_device(name: str) -> torch.device:
    """'cpu' or 'cuda' (or 'cuda:N')."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is unavailable (torch {torch.__version__})")
    return device


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


AUX_KEYS = ("aux_won", "aux_hp", "aux_mask", "aux_hp_mask", "aux_keep", "aux_keep_mask", "policy_target",
            "policy_mask")


def pop_aux(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch.pop(key) for key in AUX_KEYS if key in batch}


def predict(model: torch.nn.Module, loader: RowLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """(prediction, target) for the loader's rows, in loader order, on the CPU."""
    model.eval()
    device = next(model.parameters()).device
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("weight", None)
            pop_aux(batch)
            targets.append(batch.pop("target"))
            predictions.append(model(**to_device(batch, device)).cpu())
    return torch.cat(predictions), torch.cat(targets)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total = mask.sum()
    return (value * mask).sum() / total.clamp_min(1.0)


def aux_losses(outputs: dict[str, torch.Tensor], aux: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    won = masked_mean(F.binary_cross_entropy_with_logits(outputs["won_logit"], aux["aux_won"], reduction="none"),
                      aux["aux_mask"])
    hp = masked_mean((outputs["hp"] - aux["aux_hp"]) ** 2, aux["aux_hp_mask"])
    return won, hp


def segment_log_softmax(logits: torch.Tensor, owner: torch.Tensor, n: int) -> torch.Tensor:
    """log_softmax of each state's action logits (owner: the state of each action)."""
    maxes = logits.new_full((n,), float("-inf")).scatter_reduce(0, owner, logits.detach(), "amax")
    shifted = logits - maxes[owner]
    sums = logits.new_zeros(n).index_add(0, owner, shifted.exp())
    return shifted - sums.log()[owner]


def policy_ce(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], aux: dict[str, torch.Tensor]):
    """Per-state cross entropy against the visit distribution, and its mean over states with a policy target."""
    owner, n = batch["action_state_indices"], aux["policy_mask"].shape[0]
    logp = segment_log_softmax(outputs["policy_logits"], owner, n)
    ce = logp.new_zeros(n).index_add(0, owner, -aux["policy_target"] * logp)
    return ce, masked_mean(ce, aux["policy_mask"])


def keep_loss(outputs: dict[str, torch.Tensor], aux: dict[str, torch.Tensor]) -> torch.Tensor:
    return masked_mean((outputs["keep_fraction"] - aux["aux_keep"]) ** 2, aux["aux_keep_mask"])


def evaluate_policy(model: torch.nn.Module, loader: RowLoader) -> dict[str, float]:
    """deep_sets_v3 validation: policy cross entropy, top-1 agreement with the most visited move, and the
    potions-kept MSE, over the rows that have each target."""
    model.eval()
    device = next(model.parameters()).device
    ce_sum = agree = states = keep_sum = keep_rows = 0.0
    with torch.no_grad():
        for batch in loader:
            batch.pop("weight", None)
            batch.pop("target", None)
            aux = pop_aux(batch)
            batch = to_device(batch, device)
            aux = to_device(aux, device)
            outputs = model.forward_all(**batch)
            keep_sum += float(((outputs["keep_fraction"] - aux["aux_keep"]) ** 2 * aux["aux_keep_mask"]).sum())
            keep_rows += float(aux["aux_keep_mask"].sum())
            if "policy_logits" not in outputs:
                continue
            ce, _ = policy_ce(outputs, batch, aux)
            mask = aux["policy_mask"]
            ce_sum += float((ce * mask).sum())
            states += float(mask.sum())
            owner, n = batch["action_state_indices"], mask.shape[0]
            logits = outputs["policy_logits"]
            best_logit = logits.new_full((n,), float("-inf")).scatter_reduce(0, owner, logits, "amax")
            best_target = logits.new_full((n,), -1.0).scatter_reduce(0, owner, aux["policy_target"], "amax")
            hit = logits.new_zeros(n).index_add(0, owner, ((logits == best_logit[owner])
                                                          & (aux["policy_target"] == best_target[owner])).float())
            agree += float(((hit > 0).float() * mask).sum())
    out = {}
    if states:
        out.update(policy_validation_ce=ce_sum / states, policy_validation_top1=agree / states,
                   policy_validation_states=states)
    if keep_rows:
        out["aux_keep_validation_mse"] = keep_sum / keep_rows
    return out


def evaluate_aux(model: torch.nn.Module, loader: RowLoader) -> dict[str, float]:
    """Auxiliary validation metrics over masked rows in loader order."""
    model.eval()
    device = next(model.parameters()).device
    won_logits, hp_pred, won_targets, hp_targets, masks, hp_masks = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch.pop("weight", None)
            batch.pop("target", None)
            aux = {key: value.cpu() for key, value in pop_aux(batch).items()}
            outputs = model.forward_all(**to_device(batch, device))
            won_logits.append(outputs["won_logit"].cpu())
            hp_pred.append(outputs["hp"].cpu())
            won_targets.append(aux["aux_won"])
            hp_targets.append(aux["aux_hp"])
            masks.append(aux["aux_mask"])
            hp_masks.append(aux["aux_hp_mask"])
    logit = torch.cat(won_logits)
    prob = torch.sigmoid(logit)
    hp = torch.cat(hp_pred)
    won = torch.cat(won_targets)
    hp_target = torch.cat(hp_targets)
    mask = torch.cat(masks).bool()
    hp_mask = torch.cat(hp_masks).bool()
    metrics = {
        "aux_validation_masked_fraction": float(mask.float().mean().item()),
        "aux_hp_validation_masked_fraction": float(hp_mask.float().mean().item()),
    }
    if mask.any():
        metrics.update(
            aux_won_validation_bce=float(F.binary_cross_entropy_with_logits(logit[mask], won[mask]).item()),
            aux_won_validation_accuracy=float(((prob[mask] >= 0.5) == (won[mask] >= 0.5)).float().mean().item()),
            aux_won_validation_brier=float(((prob[mask] - won[mask]) ** 2).mean().item()),
            aux_won_validation_base_rate=float(won[mask].mean().item()),
        )
    if hp_mask.any():
        metrics["aux_hp_validation_mae"] = float((hp[hp_mask] - hp_target[hp_mask]).abs().mean().item())
    return metrics


def evaluate(model: torch.nn.Module, loader: RowLoader) -> tuple[float, float]:
    prediction, target = predict(model, loader)
    return ((prediction - target) ** 2).mean().item(), (
        prediction - target
    ).abs().mean().item()


def group_mse(rows: Rows, index: np.ndarray, prediction: torch.Tensor) -> dict[str, dict[str, float]]:
    """Per category/encounter: rows, model MSE, constant-mean MSE and teacher root_value MSE on `index`."""
    error = (prediction.numpy().astype(np.float64) - rows["target"][index]) ** 2
    target = rows["target"][index]
    teacher = (rows["root_value"][index].astype(np.float64) - target) ** 2
    keys = np.char.add(np.char.add(rows["category"][index].astype(str), "/"), rows["encounter"][index].astype(str))
    out = {}
    for key in np.unique(keys):
        mask = keys == key
        out[str(key)] = {"rows": int(mask.sum()), "mse": float(error[mask].mean()),
                         "baseline_mse": float(target[mask].var()), "teacher_mse": float(teacher[mask].mean())}
    return out


def elite_sqrt_source_weights(rows: Rows, train: np.ndarray) -> dict[str, float]:
    """Each elite gets 1/3 of loss; each source fight gets sqrt(number of decisions) mass.

    When a source is replayed, its original and replay share that mass rather than
    counting as two different decks. Validation and other experiments are unaffected.
    """
    if not np.all(rows['category'][train] == 'elite'):
        raise ValueError('elite_sqrt_source needs only elite training rows')
    episode_ids, inverse, lengths = np.unique(rows['episode_id'][train], return_inverse=True, return_counts=True)
    # Each episode has one source ID, encounter and run ID. No float conversion of large IDs.
    first = np.unique(inverse, return_index=True)[1]
    source = rows['source_episode_id'][train][first]
    source = np.array([int(s) if s is not None and not np.isnan(s) else int(e)
                       for s, e in zip(source, episode_ids)], dtype=np.int64)
    encounters = rows['encounter'][train][first]
    if len(set(zip(rows['run_id'][train], rows['episode_id'][train]))) != len(episode_ids):
        raise ValueError('episode_id collision across training runs')
    mass = np.sqrt(lengths.astype(np.float64))
    for key in np.unique(source):
        group = source == key
        if group.sum() > 1:
            mass[group] = mass[group].mean() / group.sum()
    weights = mass[inverse] / lengths[inverse]
    share = {}
    for encounter in np.unique(encounters):
        group = rows['encounter'][train] == encounter
        weights[group] *= (len(train) / len(np.unique(encounters))) / weights[group].sum()
        share[str(encounter)] = float(weights[group].sum() / len(train))
    rows.columns['weight'][train] = weights
    return share


def pinned_split(rows: Rows, reference: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """Bootstrap rows on the side the reference checkpoint put them: its train / validation episodes and
    run seeds (as row indices). Rows of other episodes are dropped."""
    split = []
    for ids, seeds in (("train_episode_ids", "train_run_seeds"), ("validation_episode_ids", "validation_run_seeds")):
        split.append(np.flatnonzero(np.isin(rows["episode_id"], reference[ids])
                                    & np.isin(rows["run_seed"], np.array(reference[seeds], dtype=np.uint64))))
    return split[0], split[1], len(rows) - len(split[0]) - len(split[1])


def load_corrections(sql, reference, v4: bool = False) -> Rows:
    """DAgger rows (apps/dagger) with target root_value (the teacher's estimate; never the actor's outcome).
    Uses the completed fights' parts even if the run itself didn't finish (each part is one whole fight).
    Every row must be a reference training episode / run seed."""
    rows = training_rows(sql, corrective=True, oracle=False, v4=v4)
    if v4:
        data_v4.assign_keep_targets(rows.columns)  # all masked: corrections carry no outcome targets
    inside = (np.isin(rows["episode_id"], reference["train_episode_ids"])
              & np.isin(rows["run_seed"], np.array(reference["train_run_seeds"], dtype=np.uint64)))
    if outside := np.unique(rows["episode_id"][~inside]).tolist():
        raise ValueError(f"corrections outside the initial checkpoint's training episodes: {outside[:5]}")
    rows.columns["target"] = rows["root_value"].astype(np.float64)
    return rows


def run(args: TrainConfig) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    architecture = args.model
    encoding_version = ENCODING_VERSIONS[architecture["kind"]]
    v4 = encoding_version == 4
    rows = training_rows(args.data, corrective=False, oracle=args.oracle, v4=v4)
    assign_targets(rows, args.label, args.blend)
    assign_aux_targets(rows)
    if v4:
        data_v4.assign_keep_targets(rows.columns)
    corrections_sql = args.corrections
    initial = args.initial_checkpoint
    weight = args.correction_weight
    split_mode = args.split
    pinned = split_mode == "pinned"
    reference = None
    corrections = None
    if initial:
        reference = json.loads(Path(initial).with_suffix(".json").read_text())
        with torch.random.fork_rng(devices=[]):  # resolved config only; leaves the seeded RNG untouched
            resolved = build(architecture).config
        if reference["architecture"] != resolved:
            raise ValueError(f"architecture {resolved} != initial checkpoint architecture {reference['architecture']}")
    if pinned:
        # Pinned: same fights on each side as the initial checkpoint; validation stays bootstrap only.
        train, valid, dropped = pinned_split(rows, reference)
        kept = np.concatenate([train, valid])
        print(f"pinned split to {initial}: dropped {dropped:,} bootstrap rows outside it", flush=True)
    else:
        if args.split == "stable":
            held = [int(r) for r in np.unique(rows["run_seed"])
                    if random.Random(int(r) ^ args.seed).random() < args.validation_fraction]
            mask = np.isin(rows["run_seed"], held)
            train, valid = np.flatnonzero(~mask), np.flatnonzero(mask)
        else:
            train, valid = split_rows(rows, args.validation_fraction, args.seed)
        kept = np.arange(len(rows))
    if not len(train) or not len(valid):
        raise ValueError("run split requires at least two run_seeds")
    if args.row_weighting == 'elite_sqrt_source':
        share = elite_sqrt_source_weights(rows, train)
        print(f'elite training loss shares: {share}', flush=True)
    train_ids = np.unique(rows["episode_id"][train]).tolist()
    valid_ids = np.unique(rows["episode_id"][valid]).tolist()
    train_runs = np.unique(rows["run_seed"][train]).tolist()
    valid_runs = np.unique(rows["run_seed"][valid]).tolist()
    row_counts = {
        column: dict(zip(*(a.tolist() for a in np.unique(rows[column][kept], return_counts=True))))
        for column in ("encounter", "category")
    }
    source_runs = sorted(set(rows["run_id"][kept]))
    bootstrap_train = train
    if corrections_sql:
        corrections = load_corrections(corrections_sql, reference, v4)
        # Per-row weights: the mean weighted loss over all training rows is
        # (1-w) * mean bootstrap loss + w * mean correction loss, whatever the row counts.
        total = len(train) + len(corrections)
        rows.columns["weight"][train] = (1 - weight) * total / len(train)
        corrections.columns["weight"][:] = weight * total / len(corrections)
        correction_ids = np.unique(corrections["episode_id"]).tolist()
        correction_runs = sorted(set(corrections["run_id"]))
        correction_count = len(corrections)
        offset = len(rows)
        rows = Rows.concat([rows, corrections])  # corrections are rows offset.. of the combined table
        corrections = np.arange(offset, len(rows))
        train = np.concatenate([train, corrections])
    aux_won_weight = args.aux_won_weight
    aux_hp_weight = args.aux_hp_weight
    aux_keep_weight = args.aux_keep_weight or 0.0
    policy_weight = args.policy_weight or 0.0
    use_aux = aux_won_weight > 0 or aux_hp_weight > 0 or v4
    train_loader = RowLoader(rows, train, args.batch_size, shuffle=True, weight=True, aux=use_aux,
                             generator=torch.Generator().manual_seed(args.seed))
    valid_loader = RowLoader(rows, valid, args.batch_size, aux=use_aux)
    model = build(architecture)
    if use_aux and not (isinstance(model, DeepSetsV3) or isinstance(model, DeepSetsV2) and model.aux_heads):
        raise ValueError("aux_won_weight/aux_hp_weight require a model with aux_heads=true (for example deep_sets_v2)")
    initial_sha = None
    if initial:
        state = torch.load(initial, map_location="cpu", weights_only=False)
        if state["architecture"] != model.config:
            raise ValueError(f"initial architecture {state['architecture']} != {model.config}")
        model.load_state_dict(state["model_state"])  # weights only; the optimizer starts fresh
        initial_sha = hashlib.sha256(Path(initial).read_bytes()).hexdigest()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_schedule = args.lr_schedule
    keep = args.keep
    # cosine: per-step decay from lr to 0 over all epochs
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
                 if lr_schedule == "cosine" else None)
    targets = rows["target"]
    baseline = float(np.mean(targets[bootstrap_train]))
    component_loaders = {
        name: RowLoader(rows, part, args.batch_size)
        for name, part in (("bootstrap", bootstrap_train), ("correction", corrections)) if corrections is not None
    }
    # The teacher's own estimate as a predictor of the label: the bar the net should approach.
    teacher_mse = float(np.mean((rows["root_value"][valid].astype(np.float64) - targets[valid]) ** 2))
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    git_dirty = (
        subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
    )

    def save_checkpoint(epoch: int, metrics: dict[str, float]) -> dict[str, Any]:
        checkpoint = {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "architecture": model.config,
            "optimizer_config": {
                "name": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "training_config": dataclasses.asdict(args),
            "encoding_version": encoding_version,
            "data_schema": "combat_v4" if all(r.startswith("combat_v4/") for r in source_runs) else combat_v3.NAME,
            "row_counts": row_counts,
            "target_name": args.label,
            "target_blend": args.blend,
            "aux_won_weight": aux_won_weight,
            "aux_hp_weight": aux_hp_weight,
            "aux_keep_weight": aux_keep_weight,
            "policy_weight": policy_weight,
            "aux_targets": {
                "keep": "MSE against clipped final potions / potions held, masked to won decision rows holding one "
                        "after the last random move (deep_sets_v3)",
                "policy": "cross entropy against the root visit share, identical tokens merged; decision rows with "
                          "visits, not oracle (deep_sets_v3)",
                "won": "BCEWithLogits against won, masked to decision rows after the last random move",
                "hp": "MSE against clipped final_hp / starting_max_hp, masked to won decision rows after the last random move",
            },
            "data_query": args.data,
            "source_runs": source_runs,
            "manifest": [json.loads((run_dir(r) / "run.json").read_text()) for r in source_runs],
            "project_git_revision": git_revision,
            "project_git_dirty": git_dirty,
            "split_mode": "pinned" if pinned else "fresh",
            "train_episode_ids": train_ids,
            "validation_episode_ids": valid_ids,
            "train_run_seeds": train_runs,
            "validation_run_seeds": valid_runs,
            "epoch": epoch,
            "metrics": metrics,
            "lr_schedule": lr_schedule,
            "keep": keep,
            "history": history,
        }
        if initial:
            checkpoint.update(
                initial_checkpoint=str(initial),
                initial_checkpoint_sha256=initial_sha,
                initialization="model weights from initial_checkpoint; fresh optimizer",
                split="pinned to initial_checkpoint train/validation episodes and run seeds" if pinned
                else "fresh episode_split(validation_fraction, seed) of the queried rows",
            )
        if corrections is not None:
            checkpoint.update(
                training_kind="corrective fine-tune (not a pure data ablation)",
                correction_weight=weight,
                correction_target="root_value (teacher_root_only)",
                correction_rows=correction_count,
                correction_episode_ids=correction_ids,
                correction_query=corrections_sql,
                correction_source=correction_runs,
                validation="bootstrap only, original targets, unweighted",
            )
        atomic_save(checkpoint, args.output)
        atomic_json(
            {key: checkpoint[key] for key in checkpoint if key != "model_state"},
            args.output.with_suffix(".json"),
        )
        return checkpoint

    print("\nDataset", flush=True)
    print("-------", flush=True)
    print(f"rows: train={len(train):,}  valid={len(valid):,}", flush=True)
    print(f"runs: train={len(train_runs):,}  valid={len(valid_runs):,}", flush=True)
    print(f"fights: train={len(train_ids):,}  valid={len(valid_ids):,}", flush=True)
    for column, counts in row_counts.items():
        print(f"{column}: {counts}", flush=True)
    print(f"target: label={args.label}  blend={args.blend}", flush=True)
    print(f"aux: won_weight={aux_won_weight:g}  hp_weight={aux_hp_weight:g}  "
          f"masked_rows={float(rows['aux_mask'][kept].mean()):.1%}  hp_masked_rows={float(rows['aux_hp_mask'][kept].mean()):.1%}",
          flush=True)
    if initial:
        print(f"initial checkpoint: {initial} (weights only, fresh optimizer; split {split_mode})", flush=True)
    if corrections is not None:
        print(f"corrections: {correction_count:,} rows, {len(correction_ids)} fights, "
              f"target root_value, weight {weight}", flush=True)

    print("\nTraining", flush=True)
    print("--------", flush=True)
    print("Each epoch prints batch progress, then validation metrics.", flush=True)
    print(
        "epoch  time    train_mse  train_mae  valid_mse  valid_mae  baseline   teacher",
        flush=True,
    )
    print(
        "-----  ------  ---------  ---------  ---------  ---------  ---------  ---------",
        flush=True,
    )
    metrics = {}
    checkpoint = {}
    history = []  # per-epoch metrics (incl. per-encounter validation MSE), also in the checkpoint json
    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        batch_count = len(train_loader)
        progress_every = max(1, batch_count // 20)
        value_loss_sum = aux_won_loss_sum = aux_hp_loss_sum = keep_loss_sum = policy_loss_sum = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            target = batch.pop("target")
            weights = batch.pop("weight")
            aux = pop_aux(batch)
            optimizer.zero_grad()
            if use_aux:
                outputs = model.forward_all(**batch)
                prediction = outputs["value"]
                aux_won_loss, aux_hp_loss = aux_losses(outputs, aux)
            else:
                prediction = model(**batch)
                aux_won_loss = prediction.new_tensor(0.0)
                aux_hp_loss = prediction.new_tensor(0.0)
            aux_keep_loss = keep_loss(outputs, aux) if v4 else prediction.new_tensor(0.0)
            policy_loss = (policy_ce(outputs, batch, aux)[1] if v4 and "policy_logits" in outputs
                           else prediction.new_tensor(0.0))
            value_loss = (weights * (prediction - target) ** 2).mean()
            loss = (value_loss + aux_won_weight * aux_won_loss + aux_hp_weight * aux_hp_loss
                    + aux_keep_weight * aux_keep_loss + policy_weight * policy_loss)
            keep_loss_sum += float(aux_keep_loss.detach().cpu())
            policy_loss_sum += float(policy_loss.detach().cpu())
            value_loss_sum += float(value_loss.detach().cpu())
            aux_won_loss_sum += float(aux_won_loss.detach().cpu())
            aux_hp_loss_sum += float(aux_hp_loss.detach().cpu())
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if batch_index == batch_count or batch_index % progress_every == 0:
                filled = round(24 * batch_index / batch_count)
                bar = "#" * filled + "." * (24 - filled)
                print(
                    f"  epoch {epoch:>3}/{args.epochs:<3} [{bar}] "
                    f"{batch_index:>5}/{batch_count:<5} loss={loss.item():.6f}",
                    flush=True,
                )
        train_value_loss = value_loss_sum / batch_count
        train_aux_won_loss = aux_won_loss_sum / batch_count
        train_aux_hp_loss = aux_hp_loss_sum / batch_count
        train_mse, train_mae = evaluate(model, train_loader)
        valid_prediction, valid_target = predict(model, valid_loader)
        valid_mse = ((valid_prediction - valid_target) ** 2).mean().item()
        valid_mae = (valid_prediction - valid_target).abs().mean().item()
        baseline_mse = float(np.mean((targets[valid] - baseline) ** 2))
        metrics = {
            "train_mse": train_mse,
            "train_mae": train_mae,
            "validation_mse": valid_mse,
            "validation_mae": valid_mae,
            "baseline_validation_mse": baseline_mse,
            "teacher_root_value_validation_mse": teacher_mse,
            "train_value_loss": train_value_loss,
            "row_weighting": args.row_weighting,
            "train_aux_won_bce": train_aux_won_loss,
            "train_aux_hp_mse": train_aux_hp_loss,
        }
        if v4:
            metrics.update(train_aux_keep_mse=keep_loss_sum / batch_count, train_policy_ce=policy_loss_sum / batch_count)
            metrics.update(evaluate_policy(model, valid_loader))
            print("  v3: " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()
                                      if k.startswith(("policy_", "aux_keep", "train_policy", "train_aux_keep"))),
                  flush=True)
        if use_aux:
            metrics.update(evaluate_aux(model, valid_loader))
        if corrections is not None:
            components = {name: evaluate(model, loader)[0] for name, loader in component_loaders.items()}
            metrics.update(
                train_bootstrap_mse=components["bootstrap"],
                train_correction_mse=components["correction"],
                train_objective=(1 - weight) * components["bootstrap"] + weight * components["correction"],
            )
            print(f"  train components: bootstrap_mse={components['bootstrap']:.6f} "
                  f"correction_mse={components['correction']:.6f} (weight {weight})", flush=True)
        elapsed = time.monotonic() - started
        print(
            "  summary "
            f"epoch={epoch}/{args.epochs} "
            f"time={elapsed:.1f}s "
            f"train_mse={train_mse:.6f} "
            f"train_mae={train_mae:.6f} "
            f"valid_mse={valid_mse:.6f} "
            f"valid_mae={valid_mae:.6f} "
            f"baseline_mse={baseline_mse:.6f} "
            f"teacher_mse={teacher_mse:.6f} "
            f"value_loss={train_value_loss:.6f} "
            f"aux_won_bce={train_aux_won_loss:.6f} "
            f"aux_hp_mse={train_aux_hp_loss:.6f} "
            f"lr_end={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        groups = group_mse(rows, valid, valid_prediction)
        print(f"  {'validation by encounter':<32}{'rows':>8}{'mse':>10}{'baseline':>10}{'teacher':>10}", flush=True)
        for name, g in groups.items():
            print(f"  {name:<32}{g['rows']:>8}{g['mse']:>10.5f}{g['baseline_mse']:>10.5f}{g['teacher_mse']:>10.5f}",
                  flush=True)
        history.append({"epoch": epoch, "lr_end": optimizer.param_groups[0]["lr"], **metrics,
                        "validation_by_encounter": groups})
        metrics = {**metrics, "validation_by_encounter": groups}
        atomic_json(history, args.output.with_name("training_history.json"))
        if keep == "last" or valid_mse < best_mse:
            best_mse = min(best_mse, valid_mse)
            checkpoint = save_checkpoint(epoch, metrics)
            print(f"  saved {args.output} (epoch {epoch})", flush=True)
        else:
            print(f"  not saved: valid_mse {valid_mse:.6f} >= best {best_mse:.6f} (keep = best)", flush=True)
    return checkpoint

