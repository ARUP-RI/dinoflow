import os
import logging
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader
from contextlib import nullcontext

from dinoflow.data import TubeData
from dinoflow.models import BTMTubes
from dinoflow.evaluate_multi_model import build_core_model_from_hparams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer(pretty_exceptions_show_locals=False)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]   %(levelname)s   %(message)s",
)
logger = logging.getLogger(__name__)


def _accessions_from_rowdict(rowdict: dict, batch_size: int) -> list[str]:
    """
    Normalize batched ACCESSION values from default_collate (tensor, ndarray, list, or str).
    Iterating a bare str would yield characters — avoid that.
    """
    acc = rowdict.get("ACCESSION")
    if acc is None:
        acc = rowdict.get("accession") or rowdict.get("Accession")
    if acc is None:
        raise KeyError(
            "Batch rowdict has no ACCESSION. Ensure the CSV has an ACCESSION column "
            "and use a recent dinoflow.data.TubeData that copies it into each sample."
        )

    if isinstance(acc, str):
        if batch_size > 1:
            raise ValueError(
                "ACCESSION is a single string for a multi-row batch; expected a list or tensor."
            )
        return [acc]

    if torch.is_tensor(acc):
        flat = acc.detach().cpu().reshape(-1)
        return [str(x.item()) for x in flat]

    if isinstance(acc, np.ndarray):
        flat = acc.reshape(-1)
        return [str(x) for x in flat.tolist()]

    if isinstance(acc, (list, tuple)):
        out = []
        for x in acc:
            if torch.is_tensor(x):
                x = x.item()
            out.append(str(x).strip() if x is not None else "")
        return out

    if np.isscalar(acc):
        return [str(acc)]

    raise TypeError(f"Unexpected ACCESSION type: {type(acc)}")


def _to_plain_dict(hp):
    """Lightning may save OmegaConf or AttributeDict hyper_parameters."""
    if hp is None:
        return {}
    if isinstance(hp, dict):
        return hp
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(hp):
            return OmegaConf.to_container(hp, resolve=True)
    except Exception:
        pass
    if hasattr(hp, "__dict__"):
        return dict(vars(hp))
    return dict(hp)


def _strip_core_state_dict(state_dict: dict) -> dict:
    """
    Map Lightning MultiTaskClassificationModel weights -> FlowMultiTaskModel keys
    (drop 'module.' DDP prefix and outer 'model.' from self.model).
    """
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        else:
            continue
        out[nk] = v
    return out


def _strip_legacy_btm_state_dict(state_dict: dict) -> dict:
    """Strip prefixes from raw BTMTubes / nested checkpoints."""
    clean_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        clean_state[nk] = v
    return clean_state


def _is_multitask_checkpoint(ckpt: dict) -> bool:
    hp = _to_plain_dict(ckpt.get("hyper_parameters"))
    return isinstance(hp, dict) and "task_defs" in hp and hp["task_defs"]


@app.command()
def compute_btm_embeddings(
    ckpt_path: str = typer.Option(..., help="Path to BTMTubes or Lightning multitask .ckpt"),
    input_csv: str = typer.Option(..., help="CSV with columns: path, ACCESSION, and labelkey column"),
    output_dir: str = typer.Option(..., help="Directory to save embeddings"),
    dataroot: str = typer.Option(".", help="Root directory prepended to CSV 'path'"),
    events: int = typer.Option(4096, help="Events to subsample per tube"),
    batch_size: int = typer.Option(16, help="Batch size"),
    num_workers: int = typer.Option(4, help="Dataloader workers"),
    labelkey: str = typer.Option("label", help="Column in CSV used as dummy label (must exist)"),
    save_mode: str = typer.Option("per", help="'per' or 'big'"),
    legacy_btm: bool = typer.Option(
        False,
        help="Force raw BTMTubes loading (ignore multitask hyper_parameters in ckpt)",
    ),
    embedding_space: str = typer.Option(
        "fused",
        help="For multitask ckpt: 'fused' = BTMTubes encoder output (matches old script); "
        "'trunk' = encoder then shared trunk MLP (same space as task heads).",
    ),
):
    """
    Emit B/T/M embeddings. Supports:

    - Legacy: weights for `BTMTubes` only (use --legacy-btm).
    - Multitask: Lightning checkpoint from `MultiTaskClassificationModel` / `FlowMultiTaskModel`
      (auto-detected via `hyper_parameters.task_defs`).
    """
    assert save_mode in ("per", "big"), "save_mode must be 'per' or 'big'"
    assert embedding_space in ("fused", "trunk"), "embedding_space must be 'fused' or 'trunk'"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    state = ckpt["state_dict"]

    use_multitask = _is_multitask_checkpoint(ckpt) and not legacy_btm

    # Minimal task_specs so TubeData runs; only path/tubes matter for embedding emit.
    task_specs = {"_emit": {"type": "bce", "out_dim": 1}}
    label_keys = {"_emit": labelkey}

    if use_multitask:
        hparams = _to_plain_dict(ckpt.get("hyper_parameters", {}))
        core = build_core_model_from_hparams(hparams, state)
        stripped = _strip_core_state_dict(state)
        missing, unexpected = core.load_state_dict(stripped, strict=False)
        logger.info(
            "Loaded FlowMultiTaskModel from Lightning checkpoint "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing[:5]:
            logger.info(f"First missing keys: {missing[:5]}")
        core.eval().to(DEVICE)
        model = core
        logger.info(f"Embedding space: {embedding_space}")
    else:
        # Legacy: instantiate BTMTubes — dims should match the trained encoder.
        model = BTMTubes(
            num_features=13,
            model_embed_dim=256,
            backbone_heads=4,
            backbone_layers=6,
            output_classes=1,
            d_ff=1024,
            include_classifier=False,
            layer_type="swiglu",
            dropout=0.1,
        )
        clean_state = _strip_legacy_btm_state_dict(state)
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        logger.info("Loaded BTMTubes checkpoint weights (legacy path)")
        logger.info(f"Missing keys: {len(missing)}")
        logger.info(f"Unexpected keys: {len(unexpected)}")
        model.eval().to(DEVICE)
        embedding_space = "fused"

    logger.info("Model loaded successfully")

    dataset = TubeData(
        input_csv,
        task_specs=task_specs,
        label_keys=label_keys,
        data_root=dataroot,
        tubes_to_return=["b", "t", "m"],
        events_to_return=events,
        labelkey=labelkey,
        report_key=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    all_embeddings = []
    all_accessions = []

    use_cuda_autocast = DEVICE.type == "cuda" and torch.cuda.is_available()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda_autocast
        else nullcontext()
    )

    with torch.inference_mode():
        for batch_idx, (batch, rowdict) in enumerate(dataloader):
            batch = {
                k: v.float().to(DEVICE, non_blocking=True)
                for k, v in batch.items()
            }

            with autocast_ctx:
                if use_multitask:
                    if embedding_space == "trunk":
                        fused = model.encoder(batch)
                        embeds = model.trunk(fused)
                    else:
                        embeds = model.encoder(batch)
                else:
                    embeds = model(batch)

            embeds = embeds.float().cpu()

            accessions = _accessions_from_rowdict(rowdict, embeds.shape[0])
            if len(accessions) != embeds.shape[0]:
                raise ValueError(
                    f"ACCESSION count ({len(accessions)}) != batch embedding rows ({embeds.shape[0]})"
                )

            if save_mode == "per":
                for i, acc in enumerate(accessions):
                    out = {
                        "embeddings": embeds[i],
                        "type": "btm_fused" if embedding_space == "fused" else "btm_trunk",
                        "embedding_space": embedding_space,
                        "multitask": use_multitask,
                    }
                    torch.save(
                        out,
                        os.path.join(output_dir, f"{acc}_btm_embeds.pt"),
                    )
            else:
                all_embeddings.append(embeds)
                all_accessions.extend(accessions)

            if batch_idx == 0:
                logger.info(f"Embedding shape: {embeds.shape}")

            if batch_idx % 20 == 0:
                logger.info(f"Processed {batch_idx * batch_size} samples")

    if save_mode == "big":
        Z = torch.cat(all_embeddings, dim=0)
        out_path = os.path.join(output_dir, "ALL_btm_preproj_embeds.pt")
        torch.save(
            {
                "ACCESSION": all_accessions,
                "embeddings": Z,
                "embedding_space": embedding_space,
                "multitask": use_multitask,
            },
            out_path,
        )
        logger.info(f"Saved {len(all_accessions)} embeddings to {out_path}")

    logger.info("Embedding generation complete.")


if __name__ == "__main__":
    app()
