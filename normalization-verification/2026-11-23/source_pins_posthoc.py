"""Source pins for statements first needed after the probes ran (labelled post hoc).

    python -I source_pins_posthoc.py

Same format as source_pins.py; writes evidence/source_pins_posthoc.json. It also
checks that the cloned LeRobot environments (deviations D7/D8) carry byte-identical
lerobot sources to the phase-2 environments for every pinned file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import source_pins as sp  # noqa: E402

ENV = Path("/home/vx/.cache/kinema-normverify")
sp.ROOTS.update({
    "lerobot033_clone": ENV / "env-lerobot033/lib/python3.11/site-packages/lerobot",
    "lerobot040_clone": ENV / "env-lerobot040/lib/python3.11/site-packages/lerobot",
})

PINS = [
    ("lr033.action_feature_exact_name", "lerobot033", "configs/policies.py", "if ft.type is FeatureType.ACTION and ft_name == ACTION:",
     "a policy's action feature must be named exactly `action`; `actions` typed ACTION is not selected"),
    ("lr033.state_feature_exact_name", "lerobot033", "configs/policies.py", "if ft.type is FeatureType.STATE and ft_name == OBS_STATE:",
     "a policy's robot state feature must be named exactly `observation.state`"),
    ("lr033.act_uses_action_feature", "lerobot033", "policies/act/modeling_act.py", "self.config.action_feature.shape[0],",
     "ACT construction dereferences action_feature"),
    ("lr033.diffusion_uses_state_feature", "lerobot033", "policies/diffusion/modeling_diffusion.py", "global_cond_dim = self.config.robot_state_feature.shape[0]",
     "diffusion construction dereferences robot_state_feature"),
    ("lr040.v20_not_implemented", "lerobot040", "datasets/backward_compatibility.py", 'raise NotImplementedError(',
     "0.4.0 offers a conversion message only for v2.1; v2.0 raises NotImplementedError"),
    ("fork.compute_norm_stats_reshape", "openpi-vlabench", "scripts/compute_norm_stats.py", "stats[key].update(values.reshape(-1, values.shape[-1]))",
     "the fork's compute_norm_stats feeds 2-D batches (deviation D3)"),
    ("fork.vlabench_inputs_pad", "openpi-vlabench", "src/openpi/policies/vlabench_policy.py", "state = transforms.pad_to_dim(ee_state, self.action_dim)",
     "VLABenchInputs pads state to the model action_dim (no-op for PI0-FAST's 7; deviation D6)"),
    ("upstream.running_stats_reshape", "openpi", "src/openpi/shared/normalize.py", "batch = batch.reshape(-1, batch.shape[-1])",
     "upstream RunningStats.update reshapes internally"),
    ("fork.normalize_no_slicing", "openpi-vlabench", "src/openpi/transforms.py", "return (x - stats.mean) / (stats.std + 1e-6)",
     "fork z-score does not slice stats to the input width (upstream does), so stats and inputs must have equal width"),
    ("upstream.normalize_slices_stats", "openpi", "src/openpi/transforms.py", "mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]",
     "upstream z-score slices stats to the input width"),
    ("gr00t.meanstd_zero_std", "gr00t", "gr00t/data/utils.py", "mask = std_vals != 0",
     "GR00T mean/std normalization leaves exactly-zero-std channels unnormalized"),
]


def main() -> None:
    out = []
    for pin_id, root, rel, anchor, claim in PINS:
        data = sp.read_source(root, rel)
        lines = data.decode().splitlines()
        hits = [i + 1 for i, line in enumerate(lines) if anchor in line]
        if not hits:
            raise SystemExit(f"anchor not found for {pin_id}")
        out.append({"id": pin_id, "root": root, "path": rel, "sha256": common.sha256_bytes(data), "lines": hits,
                    "anchor": anchor, "line_text": lines[hits[0] - 1].strip(), "claim": claim, "posthoc": True})
    clones = []
    for orig, clone in (("lerobot033", "lerobot033_clone"), ("lerobot040", "lerobot040_clone")):
        files = sorted(p.relative_to(sp.ROOTS[orig]).as_posix() for p in sp.ROOTS[orig].rglob("*.py"))
        diff = [f for f in files if common.sha256_file(sp.ROOTS[orig] / f) != common.sha256_file(sp.ROOTS[clone] / f)]
        clones.append({"original": str(sp.ROOTS[orig]), "clone": str(sp.ROOTS[clone]), "py_files": len(files), "differing": diff})
    common.write_json(common.EVIDENCE / "source_pins_posthoc.json",
                      {"schema": "kinema.normverify.source_pins/1", "posthoc": True, "pins": out, "lerobot_clone_check": clones})
    print(len(out), clones)


if __name__ == "__main__":
    main()
