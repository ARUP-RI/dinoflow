import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

IGNORE_INDEX = -100


def _is_missing(val):
    return pd.isna(val) or (isinstance(val, str) and val.strip() == "")


class TubeData(Dataset):
    """
    Expects:
      - label_keys: dict task -> CSV column name
      - task_specs: dict task -> spec, where spec includes:
          type: "bce" | "ce" | "ordinal" | "reg"
          out_dim: int
          (optional) classes: list[str] for "ce" tasks if CSV stores strings
    Produces:
      rowinfo["labels"][task] : Tensor
      rowinfo["label_masks"][task] : float Tensor {0,1}
    """

    def __init__(
        self,
        labelcsv,
        task_specs: dict,          # <- from cfg["model"]["tasks"] OR derived task_defs
        label_keys: dict,          # <- task -> csv col (from label_col mapping)
        events_to_return=-1,
        data_root="/",
        tubes_to_return=("b", "t", "m"),
        labelkey="label",          # optional legacy single label
        textroot=None,
        report_key=None,
        transforms=None,
    ):
        if isinstance(labelcsv, str):
            self.data = pd.read_csv(labelcsv)
        else:
            self.data = labelcsv

        self.task_specs = task_specs
        self.label_keys = label_keys

        self.tubes_to_return = list(tubes_to_return)
        self.dataroot = Path(data_root)
        self.events_to_return = int(events_to_return) if events_to_return is not None else -1
        self.labelkey = labelkey
        self.transforms = transforms

        self.report_key = report_key
        self.text_root = Path(textroot) if textroot is not None else None

        # sanity
        assert "path" in self.data.columns, "CSV must contain a 'path' column"
        for task, col in self.label_keys.items():
            if col not in self.data.columns:
                raise ValueError(f"CSV missing label col '{col}' for task '{task}'")

        if self.report_key is not None:
            if self.report_key not in self.data.columns:
                raise ValueError(f"{self.report_key} missing in CSV")
            if self.text_root is None:
                raise ValueError("textroot must be set if report_key is not None")

        # build class maps for CE tasks if classes provided
        self.class_map = {}
        for task, spec in self.task_specs.items():
            if spec.get("type") == "ce" and "classes" in spec:
                self.class_map[task] = {
                    str(c).strip().lower(): i for i, c in enumerate(spec["classes"])
                }

    def __len__(self):
        return len(self.data)

    def _encode(self, task: str, val):
        spec = self.task_specs.get(task, {})
        typ = spec.get("type", "bce")

        # missing
        if _is_missing(val):
            if typ in ("ce", "ordinal"):
                return IGNORE_INDEX, 0.0
            else:  # bce/reg
                return float("nan"), 0.0

        # binary bce
        if typ == "bce":
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("1", "true", "t", "yes", "y", "pos", "positive"):
                    return 1.0, 1.0
                if s in ("0", "false", "f", "no", "n", "neg", "negative"):
                    return 0.0, 1.0
                # allow "malignant" etc only if user mistakenly put strings in a bce column
                raise ValueError(f"Non-binary string '{val}' in BCE task '{task}'")
            return float(int(val)), 1.0

        # regression
        if typ == "reg":
            from dinoflow.util import reg_physical_to_training_target as _reg_map

            return _reg_map(float(val), spec), 1.0

        # ordinal
        if typ == "ordinal":
            # accept numeric or numeric string (0/1/2)
            if isinstance(val, str):
                val = int(float(val.strip()))
            return int(val), 1.0

        # ce loss/multiclass
        if typ == "ce":
            # if already numeric
            if isinstance(val, (int, np.integer)):
                return int(val), 1.0
            if isinstance(val, (float, np.floating)) and float(val).is_integer():
                return int(val), 1.0

            # else map string using classes
            s = str(val).strip().lower()

            # treat "none" as "exclude from loss/metrics"
            if s == "none":
                return IGNORE_INDEX, 0.0

            if task not in self.class_map:
                raise ValueError(
                    f"Task '{task}' is CE but no classes provided in task_specs and value is non-numeric: '{val}'. "
                    f"Either provide spec['classes'] or store numeric ids in the CSV."
                )
            if s not in self.class_map[task]:
                raise ValueError(
                    f"Unknown class '{val}' for task '{task}'. Allowed: {list(self.class_map[task].keys())}"
                )
            return int(self.class_map[task][s]), 1.0

        raise ValueError(f"Unknown task type '{typ}' for task '{task}'")

    def __getitem__(self, i):
        row = self.data.iloc[i]
        tubedata = torch.load(self.dataroot / row["path"], map_location="cpu", weights_only=False)

        tubes = {}
        for tube in self.tubes_to_return:
            x = tubedata[tube]

            if self.events_to_return != -1:
                if x.shape[0] < self.events_to_return:
                    num_repeats = self.events_to_return // x.shape[0] + 1
                    x = x.repeat(num_repeats, 1)[: self.events_to_return]
                x = subsample_events(x, int(self.events_to_return)) 

            if self.transforms is not None:
                x = self.transforms(x)

            tubes[tube] = x

        rowdict = row.to_dict()

        # multitask labels
        labels = {}
        label_masks = {}
        for task, col in self.label_keys.items():
            y, m = self._encode(task, row[col])
            typ = self.task_specs.get(task, {}).get("type", "bce")

            if typ in ("ce", "ordinal"):
                labels[task] = torch.tensor(y, dtype=torch.long)
            else:
                labels[task] = torch.tensor(y, dtype=torch.float32)

            label_masks[task] = torch.tensor(m, dtype=torch.float32)

        rowdict["labels"] = labels
        rowdict["label_masks"] = label_masks
        rowdict.update(labels)  # optional flatten

        # optional legacy single-label
        if self.labelkey in self.data.columns:
            y, m = self._encode(self.labelkey, row[self.labelkey])
            rowdict["label"] = torch.tensor(y, dtype=torch.float32)
            rowdict["label_mask"] = torch.tensor(m, dtype=torch.float32)

        # optional text embeddings
        if self.report_key is not None:
            txt_path = row[self.report_key]
            if isinstance(txt_path, str) and len(txt_path) > 0:
                full_txt_path = self.text_root / txt_path
                text_emb = torch.load(full_txt_path, map_location="cpu")
                if isinstance(text_emb, np.ndarray):
                    text_emb = torch.from_numpy(text_emb)
                rowdict["text_emb"] = text_emb.float()
            else:
                rowdict["text_emb"] = None

        return tubes, rowdict