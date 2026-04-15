"""
Evaluate MultiTaskModel checkpoints trained with train3tubesmulti

Purpose
-------
Loads a PyTorch Lightning '.ckpt' produced by multitask training (e.g. 'train3tubesmulti'),
rebuilds 'FlowMultiTaskModel' + 'MultiTaskClassificationModel' from 'hyper_parameters',
runs **per-accession** inference via 'TubeData', and writes long-form predictions plus
per-task metric tables.

Reviewer map (main pieces)
--------------------------
1. **Constants** — 'EVAL_TASK_NAMES' selects which tasks to score; 'TASK_DEFS' is the
   fallback when checkpoint 'hparams["task_defs"]' cannot be reused (label columns / schema).
2. **Checkpoint → nn.Module** — 'build_model_from_checkpoint' → 'build_core_model_from_hparams'
   reconstructs the encoder ('BTMTubes') and task heads; 'MultiTaskInferenceWrapper'
   normalizes forward output to 'dict[task, logits]'.
3. **Data + labels** — 'evaluate_multitask_model' builds 'TubeData' with tubes
   'b', 't', 'm'. 'extract_labels_and_masks' aligns masks with training
   (see 'bce_mask_matches_training' for string BCE tasks).
4. **Metrics** — 'compute_*_metrics' mirror common train/val reporting (ROC/AUPRC for BCE,
   macro F1 for CE, MAE/RMSE for regression; optional physical space via 'util.reg_*').

Outputs 
---------------------------
- '{output_prefix}_predictions_long.csv' — one row per (accession, task); BCE includes
  mean/std of probability and logit across subsamples; 'label_mask' marks excluded labels.
- '{output_prefix}_metrics_by_task.csv' — aggregated metrics per task (masked / valid rows only).

Extension for new heads
------------------------------
- Add task names to 'EVAL_TASK_NAMES' (tuple of strings).
- Extend 'TASK_DEFS' or pass '--tasks-json' with the same schema as training
  ('type', 'out_dim', 'label_col', plus CE/reg extras as in 'plmodels').
- If the core forward path differs, adjust 'MultiTaskInferenceWrapper.forward'.
"""
import os
import json
import ast
import logging
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Project imports (adjust if needed)
from dinoflow.data import TubeData
from dinoflow.models import BTMTubes, FlowMultiTaskModel
from dinoflow.plmodels import MultiTaskClassificationModel
from dinoflow.util import reg_reports_physical_metrics, reg_training_to_physical


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default heads to report in evaluate_multitask_model (order preserved in metrics tables).
# May list every trained head or a subset (e.g. action_required only).
EVAL_TASK_NAMES = ("action_required",)
TASK_DEFS = {
    "action_required": {
        "type": "bce",
        "out_dim": 1,
        "label_col": "ACTION_REQUIRED",
    }
    #"suboptimal_viability": {
        #"type": "reg",
        #"out_dim": 1,
        #"label_col": "viability_pct_y",
        #"loss": "huber",
        #"reg_target_transform": "logit",
        #"logit_max": 100.0,
        #"logit_eps": 1.0e-4,
    #},
    #"reactive_vs_malignant": {
    #    "type": "bce",
    #    "out_dim": 1,
    #    "label_col": "malignant_vs_reactive",
    #},
}

def _task_defs_from_hparams(hparams, names: tuple[str, ...]) -> dict | None:
    """
    Build task_defs for eval using checkpoint hyper_parameters.task_defs when present,
    restricted to `names`, so label_col matches training.
    """
    if hparams is None:
        return None
    raw_td = hparams["task_defs"] if isinstance(hparams, dict) else getattr(hparams, "task_defs", None)
    if raw_td is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(raw_td):
            raw_td = OmegaConf.to_container(raw_td, resolve=True)
    except Exception:
        pass
    if not isinstance(raw_td, dict):
        return None
    out = {}
    for name in names:
        spec = raw_td.get(name)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            try:
                spec = dict(spec)
            except Exception:
                continue
        out[name] = spec
    if not out:
        return None
    return out

# Utilities
def subsample_events(x, num_events):
    """
    Select a random sample of events and return those.
    If the input has fewer events than `num_events`, all available events are used.
    """
    if len(x.shape) == 3:
        num_available = x.shape[1]
        ev = torch.randperm(num_available)[: min(num_events, num_available)]
        return x[:, ev, :]
    elif len(x.shape) == 2:
        num_available = x.shape[0]
        ev = torch.randperm(num_available)[: min(num_events, num_available)]
        return x[ev, :]
    else:
        raise ValueError(
            f"Input tensor must have 2 or 3 dimensions (found {len(x.shape)})"
        )


def safe_isna(x):
    try:
        return pd.isna(x)
    except Exception:
        return False


def parse_if_json_list(x):
    """Parse list-like CSV cell values (e.g. '[0.1,0.2,0.7]') back into Python list."""
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try:
                return json.loads(x)
            except Exception:
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return x
    return x


# Metrics
def compute_binary_metrics(
    y_true,
    y_pred_probs,
    fpr_thresholds=(0.01, 0.02, 0.05),
    sensitivity_thresholds=(0.95, 0.99, 0.995),
):
    y_true = np.asarray(y_true).astype(int)
    y_pred_probs = np.asarray(y_pred_probs).astype(float)

    metrics = {"n": int(len(y_true))}
    if len(y_true) == 0:
        metrics["warning"] = "No rows"
        return metrics

    metrics["positive_rate"] = float(y_true.mean())

    # Degenerate case: only one class present
    if len(np.unique(y_true)) < 2:
        metrics["warning"] = "Only one class present; ROC/AUC metrics undefined."
        return metrics

    # ROC / AUC
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_probs)
    metrics["roc_auc"] = float(auc(fpr, tpr))

    # recall @ FPR
    for fpr_threshold in fpr_thresholds:
        idx = fpr <= fpr_threshold
        metrics[f"recall_at_fpr_{fpr_threshold}"] = float(np.max(tpr[idx]) if np.any(idx) else 0.0)

    # specificity @ recall
    for sens_threshold in sensitivity_thresholds:
        idx = tpr >= sens_threshold
        spec_vals = 1.0 - fpr[idx]
        metrics[f"specificity_at_recall_{sens_threshold}"] = float(np.max(spec_vals) if len(spec_vals) else 0.0)

    # PR / AUPRC
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_pred_probs)
    metrics["auprc"] = float(average_precision_score(y_true, y_pred_probs))

    # F1-max threshold
    denom = precision_vals + recall_vals
    denom[denom == 0] = 1e-8
    f1_scores = 2 * precision_vals * recall_vals / denom
    max_f1_idx = int(np.argmax(f1_scores))

    threshold = thresholds_pr[max_f1_idx] if max_f1_idx < len(thresholds_pr) else 0.5
    best_precision = precision_vals[max_f1_idx]
    best_recall = recall_vals[max_f1_idx]
    best_f1 = f1_scores[max_f1_idx]

    metrics["threshold"] = float(threshold)
    metrics["precision"] = float(best_precision)
    metrics["recall"] = float(best_recall)
    metrics["fscore"] = float(best_f1)

    # Apply F1 threshold
    y_pred = (y_pred_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["accuracy"] = float((tp + tn) / max(1, tp + tn + fp + fn))
    metrics["specificity"] = float(tn / max(1, tn + fp))
    metrics["ppv"] = float(tp / max(1, tp + fp))
    metrics["npv"] = float(tn / max(1, tn + fn))

    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)

    return metrics


def compute_multiclass_metrics(y_true, y_pred_class, class_names=None):
    """
    Multiclass metrics from integer class predictions (argmax / mapped labels).

    Includes accuracy, macro/weighted F1, macro precision/recall, and per-class
    precision/recall/support when ``class_names`` aligns with class indices.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_class = np.asarray(y_pred_class).astype(int)

    if len(y_true) == 0:
        return {"n": 0, "warning": "No rows"}

    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred_class)),
        "macro_f1": float(f1_score(y_true, y_pred_class, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred_class, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred_class, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred_class, average="macro", zero_division=0)),
    }

    classes = sorted(np.unique(y_true))
    for c in classes:
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred_class == c).astype(int)

        tp = int(((y_true_c == 1) & (y_pred_c == 1)).sum())
        fp = int(((y_true_c == 0) & (y_pred_c == 1)).sum())
        fn = int(((y_true_c == 1) & (y_pred_c == 0)).sum())
        support = int(y_true_c.sum())

        cname = class_names[c] if (class_names is not None and c < len(class_names)) else str(c)
        metrics[f"class_{cname}_support"] = support
        metrics[f"class_{cname}_precision"] = float(tp / max(1, tp + fp))
        metrics[f"class_{cname}_recall"] = float(tp / max(1, tp + fn))

    return metrics


def compute_regression_metrics(y_true, y_pred):
    """Mean absolute error and RMSE in the provided target space (training or physical)."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    if len(y_true) == 0:
        return {"n": 0, "warning": "No rows"}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
    }

# Inference postprocessing
def postprocess_task_logits(logits, task_def):
    """
    Standardize outputs for one task.
    Expects logits for batch size 1.
    """
    task_type = task_def["type"]

    if task_type == "bce":
        x = logits
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)[:, 0]

        prob = torch.sigmoid(x)
        return {
            "logit": float(x[0].detach().float().cpu().item()),
            "probability": float(prob[0].detach().float().cpu().item()),
            "pred_class_0p5": int((prob[0] >= 0.5).item()),
        }

    elif task_type == "ce":
        if logits.ndim != 2:
            logits = logits.reshape(logits.shape[0], -1)
        probs = torch.softmax(logits, dim=-1)
        pred_class = int(torch.argmax(probs[0]).detach().cpu().item())
        return {
            "pred_class": pred_class,
            "pred_probs": probs[0].detach().float().cpu().numpy().tolist(),
        }

    elif task_type == "reg":
        x = logits.reshape(logits.shape[0], -1)[:, 0]
        t = float(x[0].detach().float().cpu().item())
        out = {"pred_value": t}
        if reg_reports_physical_metrics(task_def):
            out["pred_value_physical"] = float(reg_training_to_physical(t, task_def))
        return out

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

# MODEL LOADER 
class MultiTaskInferenceWrapper(torch.nn.Module):
    """
    Wraps your inference model so forward(batch) returns dict[task] = logits.
    If your loaded object already does this, you may not need this wrapper.
    """
    def __init__(self, core_model):
        super().__init__()
        self.core_model = core_model

    def forward(self, batch):
        out = self.core_model(batch)

        # Common cases:
        # 1) out is already dict task->logits
        if isinstance(out, dict):
            return out

        # 2) LightningModule where `.model(batch)` gives dict task->logits
        if hasattr(self.core_model, "model"):
            maybe = self.core_model.model(batch)
            if isinstance(maybe, dict):
                return maybe

        raise TypeError(
            "Model forward did not return dict(task->logits). "
            "Edit MultiTaskInferenceWrapper.forward() for your model."
        )


def build_model_from_checkpoint(checkpoint_path):
    """
    Rebuild MultiTaskClassificationModel + load checkpoint weights.

    You only need to implement build_core_model_from_hparams(hparams, state_dict).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint["state_dict"]

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
    logger.info(f"hparams keys: {list(hparams.keys()) if isinstance(hparams, dict) else type(hparams)}")

    # Helpful debugging
    logger.info("First 30 state_dict keys:")
    for i, (k, v) in enumerate(state_dict.items()):
        if i >= 30:
            break
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
        logger.info(f"  {k}: {shape}")

    # Build core model (YOU implement this one function)
    core_model = build_core_model_from_hparams(hparams, state_dict)

    # rebuild PL module
    # IMPORTANT: import your actual class path
    # from your_training_module import MultiTaskClassificationModel

    pl_module = MultiTaskClassificationModel(
        model=core_model,
        task_defs=hparams["task_defs"],
        task_weights=hparams["task_weights"],
        primary_task=hparams.get("primary_task", "action_required"),
        label_map=hparams.get("label_map", None),
        emit_predictions=hparams.get("emit_predictions", False),

        min_lr=hparams.get("min_lr", 1e-5),
        max_lr=hparams.get("max_lr", 2.5e-4),
        trunk_lr_mult=hparams.get("trunk_lr_mult", 0.3),
        warmup_epochs=hparams.get("warmup_epochs", 10),
        decay_end_epoch=hparams.get("decay_end_epoch", 100),
        hold_epochs=hparams.get("hold_epochs", 5),
        freeze_encoder_iters=hparams.get("freeze_encoder_iters", 0),

        checkpoint_monitor=hparams.get("checkpoint_monitor", "val/specificity_at_recall_0.99"),
        checkpoint_mode=hparams.get("checkpoint_mode", "max"),

        pos_weight=hparams.get("pos_weight", None),

        fpr_thresholds=tuple(hparams.get("fpr_thresholds", (0.01, 0.02, 0.05))),
        sensitivity_thresholds=tuple(hparams.get("sensitivity_thresholds", (0.95, 0.99, 0.995))),

        weight_decay=hparams.get("weight_decay", 1e-3),
    )

    # Load full PL module weights
    missing, unexpected = pl_module.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys ({len(missing)}): {missing[:20]}")
    logger.info(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    pl_module.eval()

    # Wrap for eval script so model(batch) returns dict(task->logits)
    model = MultiTaskInferenceWrapper(pl_module)
    return model, hparams

def build_core_model_from_hparams(hparams, state_dict):
    """
    Reconstruct ``FlowMultiTaskModel(encoder=..., task_defs=..., trunk_cfg=...)``.

    Fills missing encoder/trunk fields by inspecting ``state_dict`` key shapes so older
    checkpoints with partial ``model_conf`` / ``trunk_cfg`` still load. ``BTMTubes`` is
    built with ``include_classifier=False`` because task heads live on ``FlowMultiTaskModel``.
    """
    task_defs = hparams["task_defs"]
    model_conf = hparams.get("model_conf", {}) or {}
    trunk_cfg = hparams.get("trunk_cfg", {}) or {}

    logger.info(f"task_defs keys: {list(task_defs.keys())}")
    logger.info(f"model_conf from hparams: {model_conf}")
    logger.info(f"trunk_cfg from hparams: {trunk_cfg}")

    # Infer encoder config (fallbacks from state_dict if hparams incomplete)
    # Defaults
    num_features = int(model_conf.get("num_features", 13))
    model_embed_dim = model_conf.get("model_embed_dim", model_conf.get("model_dim", 256))
    backbone_heads = model_conf.get("backbone_heads", model_conf.get("heads", 4))
    backbone_layers = model_conf.get("backbone_layers", model_conf.get("layers", 6))
    d_ff = model_conf.get("d_ff", None)
    layer_type = model_conf.get("layer_type", None)
    output_scale_factor = float(model_conf.get("output_scale_factor", 1.0))

    # Infer layer_type from state dict if not present
    if layer_type is None:
        has_linear_gate = any("linear_gate" in k for k in state_dict.keys())
        layer_type = "swiglu" if has_linear_gate else "normal"

    # Infer d_ff from any backbone linear1 weight if missing
    if d_ff is None:
        for k, v in state_dict.items():
            if "encoder" in k and "linear1.weight" in k and hasattr(v, "shape"):
                d_ff = int(v.shape[0])
                break
        if d_ff is None:
            for k, v in state_dict.items():
                if "linear1.weight" in k and hasattr(v, "shape"):
                    d_ff = int(v.shape[0])
                    break
        if d_ff is None:
            d_ff = 2048

    # Infer trunk config from state dict if missing
    # trunk.net.0.weight shape = [hidden_dim, fused_dim]
    if "hidden_dim" not in trunk_cfg or trunk_cfg.get("hidden_dim") is None:
        for k, v in state_dict.items():
            if "model.trunk.net.0.weight" in k and hasattr(v, "shape"):
                trunk_cfg["hidden_dim"] = int(v.shape[0])
                break
            if "trunk.net.0.weight" in k and hasattr(v, "shape"):
                trunk_cfg["hidden_dim"] = int(v.shape[0])
                break
    trunk_cfg.setdefault("hidden_dim", 256)

    # Count trunk layers from linears in trunk.net.*.weight
    if "n_layers" not in trunk_cfg or trunk_cfg.get("n_layers") is None:
        linear_count = 0
        for k, v in state_dict.items():
            # each MLP hidden layer contributes one Linear weight in trunk.net.<idx>.weight
            if (k.startswith("model.trunk.net.") or k.startswith("trunk.net.")) and k.endswith(".weight") and hasattr(v, "shape"):
                if len(v.shape) == 2:
                    linear_count += 1
        if linear_count > 0:
            trunk_cfg["n_layers"] = linear_count
    trunk_cfg.setdefault("n_layers", 2)

    # Dropout and LN are not always inferable; use hparams/defaults
    trunk_cfg.setdefault("drop", 0.1)
    trunk_cfg.setdefault("use_ln", True)

    logger.info(
        f"Reconstructed encoder cfg: num_features={num_features}, embed_dim={model_embed_dim}, "
        f"heads={backbone_heads}, layers={backbone_layers}, d_ff={d_ff}, layer_type={layer_type}"
    )
    logger.info(f"Reconstructed trunk_cfg: {trunk_cfg}")

    #build encodewr (BTM tubes)
    # NOTE:
    # - include_classifier=False is key since FlowMultiTaskModel adds task heads itself
    # - output_classes is irrelevant when include_classifier=False, but some BTMTubes
    #   versions require it, so pass 1.
    encoder = BTMTubes(
        num_features=num_features,
        model_embed_dim=int(model_embed_dim),
        backbone_heads=int(backbone_heads),
        backbone_layers=int(backbone_layers),
        output_classes=1,
        d_ff=int(d_ff),
        layer_type=layer_type,
        output_scale_factor=output_scale_factor,
        include_classifier=False,
    )


    # Build core multitask model
    core_model = FlowMultiTaskModel(
        encoder=encoder,
        task_defs=task_defs,
        trunk_cfg=trunk_cfg,
    )

    # Optional: sanity-check output dims against checkpoint heads
    for task, cfg in task_defs.items():
        if task in core_model.outs:
            expected = int(cfg["out_dim"]) - 1 if cfg["type"] == "ordinal" else int(cfg["out_dim"])
            actual = core_model.outs[task].out_features
            if expected != actual:
                logger.warning(f"Task {task}: expected out_features={expected}, got {actual}")

    return core_model


def bce_mask_matches_training(task: str, raw_label) -> float:
    """
    Per-sample mask (1 = use in supervised loss / metrics) aligned with data.TubeData._encode
    for BCE tasks that use string labels including an excluded class (e.g. 'none').
    """
    if safe_isna(raw_label):
        return 0.0
    if isinstance(raw_label, str):
        s = raw_label.strip().lower()
        if task == "reactive_vs_malignant":
            if s == "none":
                return 0.0
            if s in ("malignant", "reactive"):
                return 1.0
        if task == "acute_maturation":
            if s == "none":
                return 0.0
            if s in ("mature", "acute"):
                return 1.0
    return 1.0


# Dataset label extraction (one place to tweak if needed)
def extract_labels_and_masks(row_data, rowinfo, task_defs):
    """
    Returns:
      labels_dict: task -> label
      masks_dict: task -> 0/1
    Tries rowinfo['labels']/rowinfo['label_masks'] first, then falls back to row_data[label_col].

    For reactive_vs_malignant, training uses malignant=1, reactive=0, none=masked (same as data.py).
    """
    labels_dict = {}
    masks_dict = {}

    # rowinfo may be dict or something else depending on TubeData
    if isinstance(rowinfo, dict):
        if isinstance(rowinfo.get("labels", None), dict):
            labels_dict.update(rowinfo["labels"])
        if isinstance(rowinfo.get("label_masks", None), dict):
            masks_dict.update(rowinfo["label_masks"])

    # Fallback to row_data columns
    for task, task_def in task_defs.items():
        if task not in labels_dict:
            label_col = task_def.get("label_col", task)
            labels_dict[task] = row_data.get(label_col, np.nan)

        if task not in masks_dict:
            ttype = str(task_def.get("type", "bce")).lower()
            if ttype == "bce":
                masks_dict[task] = float(bce_mask_matches_training(task, labels_dict[task]))
            else:
                masks_dict[task] = 0.0 if safe_isna(labels_dict[task]) else 1.0

    # Coerce string BCE labels to the same float targets as training (for metrics CSV)
    for task, task_def in task_defs.items():
        if str(task_def.get("type", "")).lower() != "bce":
            continue
        v = labels_dict.get(task)
        if isinstance(v, str) and task == "reactive_vs_malignant":
            s = v.strip().lower()
            if s == "malignant":
                labels_dict[task] = 1.0
            elif s == "reactive":
                labels_dict[task] = 0.0
            elif s == "none":
                labels_dict[task] = 0.0

    return labels_dict, masks_dict

# Evaluation
def evaluate_multitask_model(
    model,
    dataset_csv: str,
    dataroot: str,
    task_defs: dict,
    events_per_sample: int = 4096,
    num_subsamples: int = 1,
    batch_size: int = 1,  # kept for compatibility; loop is per-sample
    device: str = None,
    focus_accessions: list = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TubeData(
        dataset_csv,
        events_to_return=-1,
        data_root=dataroot,
        tubes_to_return=["b", "t", "m"],
        task_specs=task_defs,
        label_keys={t: cfg.get("label_col", t) for t, cfg in task_defs.items()},
        report_key=None,
        transforms=None,
    )

    if focus_accessions:
        indices = [
            i for i in range(len(dataset))
            if dataset.get_row_data(i).get("ACCESSION") in focus_accessions
        ]
    else:
        indices = range(len(dataset))

    model.eval()
    model.to(device)

    long_rows = []

    for idx in tqdm(indices, desc="Processing samples"):
        row_data = dataset.get_row_data(idx)
        accession = row_data.get("ACCESSION", f"idx_{idx}")

        sample = dataset[idx]
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            full_tubes, rowinfo = sample
        else:
            # fallback if dataset returns only tubes
            full_tubes, rowinfo = sample, {}

        # Validate event counts
        skip_sample = False
        for tube, tube_data in full_tubes.items():
            if tube_data.ndim == 3:
                n_events = tube_data.shape[1]
            elif tube_data.ndim == 2:
                n_events = tube_data.shape[0]
            else:
                raise ValueError(f"Invalid shape for tube {tube}: {tuple(tube_data.shape)}")
            if n_events < events_per_sample:
                skip_sample = True
                break

        if skip_sample:
            continue

        labels_dict, masks_dict = extract_labels_and_masks(row_data, rowinfo, task_defs)

        # Collect subsample outputs per task
        sample_task_preds = {task: [] for task in task_defs.keys()}

        for sidx in range(num_subsamples):
            subsampled_tubes = {
                tube: subsample_events(tube_data, events_per_sample)
                for tube, tube_data in full_tubes.items()
            }

            batch = {tube: data.unsqueeze(0).to(device) for tube, data in subsampled_tubes.items()}

            use_cuda_autocast = ("cuda" in str(device)) and torch.cuda.is_available()
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda_autocast else nullcontext()

            with torch.no_grad():
                with autocast_ctx:
                    out = model(batch)  # expected dict task -> logits

            if not isinstance(out, dict):
                raise TypeError(
                    f"Model output must be dict(task->logits), got {type(out)}. "
                    f"Edit wrapper/loader."
                )

            for task, task_def in task_defs.items():
                if task not in out:
                    continue
                pred = postprocess_task_logits(out[task], task_def)
                sample_task_preds[task].append(pred)

        # Aggregate across subsamples and store one row per (accession, task)
        for task, task_def in task_defs.items():
            preds = sample_task_preds.get(task, [])
            if len(preds) == 0:
                continue

            true_label = labels_dict.get(task, np.nan)
            label_mask = int(masks_dict.get(task, 0))

            row = {
                "accession": accession,
                "task": task,
                "task_type": task_def["type"],
                "true_label": true_label,
                "label_mask": label_mask,
                "num_subsamples": len(preds),
            }

            if task_def["type"] == "bce":
                probs = [p["probability"] for p in preds]
                logits = [p["logit"] for p in preds]
                row.update({
                    "mean_probability": float(np.mean(probs)),
                    "std_probability": float(np.std(probs)),
                    "mean_logit": float(np.mean(logits)),
                    "std_logit": float(np.std(logits)),
                    "pred_class_0p5": int(np.mean(probs) >= 0.5),
                })

            elif task_def["type"] == "ce":
                prob_mat = np.stack([np.array(p["pred_probs"], dtype=float) for p in preds], axis=0)
                mean_probs = prob_mat.mean(axis=0)
                row.update({
                    "pred_class": int(np.argmax(mean_probs)),
                    "pred_probs": json.dumps(mean_probs.tolist()),  # CSV-safe
                })

            elif task_def["type"] == "reg":
                vals = [p["pred_value"] for p in preds]
                row.update({
                    "pred_value_mean": float(np.mean(vals)),
                    "pred_value_std": float(np.std(vals)),
                })
                if reg_reports_physical_metrics(task_def):
                    phys = [p.get("pred_value_physical", np.nan) for p in preds]
                    row["pred_value_mean_physical"] = float(np.nanmean(phys))
                    row["pred_value_std_physical"] = float(np.nanstd(phys))

            long_rows.append(row)

    pred_long_df = pd.DataFrame(long_rows)

    # Metrics by task
    metrics_rows = []

    for task, task_def in task_defs.items():
        tdf = pred_long_df[(pred_long_df["task"] == task) & (pred_long_df["label_mask"] == 1)].copy()

        if len(tdf) == 0:
            metrics_rows.append({
                "task": task,
                "task_type": task_def["type"],
                "warning": "No valid labeled rows",
            })
            continue

        tdf = tdf[~tdf["true_label"].isna()].copy()
        if len(tdf) == 0:
            metrics_rows.append({
                "task": task,
                "task_type": task_def["type"],
                "warning": "No non-NaN labels",
            })
            continue

        # Convert CE string labels to indices if needed
        if task_def["type"] == "ce" and "class_to_idx" in task_def:
            tdf["true_label"] = tdf["true_label"].map(task_def["class_to_idx"])

        if task_def["type"] == "bce":
            y_true = tdf["true_label"].astype(int).values
            y_score = tdf["mean_probability"].astype(float).values
            m = compute_binary_metrics(y_true, y_score)
            m.update({"task": task, "task_type": "bce"})
            metrics_rows.append(m)

        elif task_def["type"] == "ce":
            y_true = tdf["true_label"].astype(int).values
            y_pred = tdf["pred_class"].astype(int).values
            class_names = task_def.get("classes", None)
            m = compute_multiclass_metrics(y_true, y_pred, class_names=class_names)
            m.update({"task": task, "task_type": "ce"})
            metrics_rows.append(m)

        elif task_def["type"] == "reg":
            y_true_t = tdf["true_label"].astype(float).values
            y_pred_t = tdf["pred_value_mean"].astype(float).values
            m = compute_regression_metrics(y_true_t, y_pred_t)
            m.update({"task": task, "task_type": "reg", "target_space": "training"})
            if reg_reports_physical_metrics(task_def):
                y_p = reg_training_to_physical(y_true_t, task_def)
                yhat_p = reg_training_to_physical(y_pred_t, task_def)
                mp = compute_regression_metrics(y_p, yhat_p)
                m["mae_physical"] = mp["mae"]
                m["rmse_physical"] = mp["rmse"]
            metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)

    return pred_long_df, metrics_df


# Pretty printing
def print_summary(pred_df, metrics_df, task_defs):
    print("=" * 80)
    print("MULTITASK MODEL EVALUATION")
    print("=" * 80)
    print(f"Rows in predictions table: {len(pred_df)}")
    print()

    if len(pred_df) == 0:
        print("No predictions generated.")
        return

    print("Per-task counts:")
    counts = pred_df.groupby("task")["accession"].count().sort_values(ascending=False)
    for task, n in counts.items():
        labeled_n = int(((pred_df["task"] == task) & (pred_df["label_mask"] == 1)).sum())
        print(f"  {task}: {n} predictions ({labeled_n} labeled)")
    print()

    print("Metrics by task:")
    for _, row in metrics_df.iterrows():
        task = row.get("task", "unknown")
        ttype = row.get("task_type", "unknown")
        print(f"- {task} ({ttype})")

        if pd.notna(row.get("warning", np.nan)):
            print(f"    warning: {row['warning']}")
            continue

        if ttype == "bce":
            # print a compact subset
            for k in [
                "n", "positive_rate", "roc_auc", "auprc",
                "accuracy", "precision", "recall", "fscore",
                "specificity", "npv", "threshold"
            ]:
                if k in row and pd.notna(row[k]):
                    print(f"    {k}: {row[k]}")
        elif ttype == "ce":
            for k in ["n", "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]:
                if k in row and pd.notna(row[k]):
                    print(f"    {k}: {row[k]}")
        elif ttype == "reg":
            for k in ["n", "mae", "rmse"]:
                if k in row and pd.notna(row[k]):
                    print(f"    {k}: {row[k]}")
        print()


def main():
    """CLI entrypoint; see module docstring for architecture and output files."""
    parser = argparse.ArgumentParser(
        description="Evaluate multitask tube model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Outputs: {prefix}_predictions_long.csv and {prefix}_metrics_by_task.csv. "
        "See module docstring in evaluate_multi_model.py for reviewer-oriented overview.",
    )
    parser.add_argument("checkpoint_path", help="Path to checkpoint (.ckpt)")
    parser.add_argument("dataset_csv", help="Path to evaluation CSV")
    parser.add_argument("--dataroot", default=".", help="Root directory for data")
    parser.add_argument("--events", type=int, default=4096, help="Events per sample per tube")
    parser.add_argument("--num-subsamples", type=int, default=1, help="Number of random subsamples per accession")
    parser.add_argument("--batch-size", type=int, default=1, help="Kept for compatibility (evaluation runs per-sample)")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output CSVs")
    parser.add_argument("--device", default=None, help="cuda / cpu (default auto)")
    parser.add_argument("--focus-accessions", nargs="+", help="Evaluate only these accessions")
    parser.add_argument(
        "--tasks-json",
        default=None,
        help="Optional path to JSON task defs to override defaults (otherwise checkpoint hparams or TASK_DEFS)",
    )
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Dataset CSV: {args.dataset_csv}")

    # Load model first so we can take task_defs from checkpoint when appropriate
    model, hparams = build_model_from_checkpoint(args.checkpoint_path)

    if args.tasks_json:
        with open(args.tasks_json, "r") as f:
            task_defs = json.load(f)
        logger.info(f"Loaded task defs from {args.tasks_json}")
    else:
        merged = _task_defs_from_hparams(hparams, EVAL_TASK_NAMES)
        if merged is not None and all(n in merged for n in EVAL_TASK_NAMES):
            task_defs = merged
            logger.info(
                "Using task_defs for %s from checkpoint hparams",
                ", ".join(EVAL_TASK_NAMES),
            )
        elif merged:
            task_defs = merged
            logger.warning(
                "Checkpoint task_defs missing some of %s; evaluating available: %s",
                ", ".join(EVAL_TASK_NAMES),
                list(merged.keys()),
            )
        else:
            task_defs = dict(TASK_DEFS)
            logger.info(
                "Using built-in TASK_DEFS for %s (no matching task_defs in checkpoint)",
                ", ".join(EVAL_TASK_NAMES),
            )

    logger.info(f"Tasks: {list(task_defs.keys())}")

    # Evaluate
    pred_df, metrics_df = evaluate_multitask_model(
        model=model,
        dataset_csv=args.dataset_csv,
        dataroot=args.dataroot,
        task_defs=task_defs,
        events_per_sample=args.events,
        num_subsamples=args.num_subsamples,
        batch_size=args.batch_size,
        device=device,
        focus_accessions=args.focus_accessions,
    )

    if len(pred_df) == 0:
        logger.error("No samples could be processed. Check event count threshold / data paths.")
        return

    # Save outputs
    pred_csv = f"{args.output_prefix}_predictions_long.csv"
    metrics_csv = f"{args.output_prefix}_metrics_by_task.csv"

    pred_df.to_csv(pred_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    logger.info(f"Saved predictions: {pred_csv}")
    logger.info(f"Saved metrics: {metrics_csv}")

    # Optional per-task BCE threshold sweeps (simple artifact)
    # You can uncomment if useful for your FP/FN analysis
    # save_bce_threshold_sweeps(pred_df, task_defs, f"{args.output_prefix}_bce_threshold_sweeps.csv")

    print_summary(pred_df, metrics_df, task_defs)


# Optional helper (commented out in main by default)
def save_bce_threshold_sweeps(pred_df, task_defs, out_csv, thresholds=None):
    """
    Write a grid of confusion-matrix-derived stats over BCE probability thresholds.

    Intended for FP/FN tradeoff analysis; not invoked from ``main`` by default.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    rows = []
    for task, task_def in task_defs.items():
        if task_def["type"] != "bce":
            continue
        tdf = pred_df[(pred_df["task"] == task) & (pred_df["label_mask"] == 1)].copy()
        if len(tdf) == 0:
            continue

        y_true = tdf["true_label"].astype(int).values
        y_score = tdf["mean_probability"].astype(float).values

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            rows.append({
                "task": task,
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": float(tp / max(1, tp + fp)),
                "recall": float(tp / max(1, tp + fn)),
                "specificity": float(tn / max(1, tn + fp)),
                "npv": float(tn / max(1, tn + fn)),
                "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"Saved BCE threshold sweeps: {out_csv}")


if __name__ == "__main__":
    main()