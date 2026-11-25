"""Deviation D9 (post hoc, exploratory, never used for a verdict).

    env-lerobot040/bin/python -I posthoc_feature_scan.py

probe_lerobot.py 0.4.0 found in-sample outputs up to ~2e9 on the full sampled
rows of r393 under QUANTILES, outside the flagged dim. This scan names the dims:
for each STATE mode it runs the same NormalizerProcessorStep on the same sampled
full rows and lists the ten dims with the largest |output| together with their
stored std and q99 - q01. It informs proposed later detector changes only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HOME=str(common.SCRATCH / "posthoc-hf"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.core import TransitionKey  # noqa: E402
from lerobot.processor.normalize_processor import NormalizerProcessorStep  # noqa: E402

CID = "r393:c7:f14"


def main() -> None:
    exp = {c["case_id"]: c for c in common.load_json(common.EXPECTATIONS)["cases"]}[CID]
    inp = {c["case_id"]: c for c in common.load_json(common.EVIDENCE / "inputs.json")["cases"]}[CID]
    ch = {c["case_id"]: c for c in common.load_json(common.EVIDENCE / "channels.json")["cases"]}[CID]
    if common.sha256_file(ch["sample_file"]["path"]) != ch["sample_file"]["sha256"]:
        raise SystemExit("sample file hash changed")
    feat = exp["finding"]["details"]["feature"]
    meta = LeRobotDatasetMetadata(exp["source"]["repo"], root=Path(inp["root"]), revision=exp["source"]["revision"])
    st = {k: np.asarray(v, dtype=np.float64) for k, v in meta.stats[feat].items()}
    rows = np.load(ch["sample_file"]["path"])["full_rows"]
    out = {"schema": "kinema.normverify.posthoc_feature_scan/1", "deviation": "D9", "case_id": CID,
           "label": "post hoc exploration; not a verdict input", "rows": int(len(rows)), "modes": {}}
    for mode in ("MEAN_STD", "MIN_MAX", "QUANTILES"):
        step = NormalizerProcessorStep(features={feat: PolicyFeature(type=FeatureType.STATE, shape=(rows.shape[1],))},
                                       norm_map={FeatureType.STATE: NormalizationMode[mode]}, stats=meta.stats)
        z = step(create_transition(observation={feat: torch.from_numpy(rows.astype(np.float32))}))[TransitionKey.OBSERVATION][feat].numpy()
        peak = np.abs(z).max(axis=0)
        top = np.argsort(-peak)[:10]
        out["modes"][mode] = {
            "dims_over_10": int((peak > 10).sum()), "dims_over_1000": int((peak > 1000).sum()),
            "top": [{"dim": int(j), "abs_max_out": float(peak[j]), "stored_std": float(st["std"][j]),
                     "stored_q_range": float(st["q99"][j] - st["q01"][j]), "stored_range": float(st["max"][j] - st["min"][j])}
                    for j in top]}
    common.write_json(common.EVIDENCE / "posthoc_feature_scan.json", out)
    print("ok")


if __name__ == "__main__":
    main()
