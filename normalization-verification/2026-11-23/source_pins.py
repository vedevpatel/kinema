"""Record the exact source lines that each source-inspection statement relies on.

Run with any Python 3.11+: `python -I source_pins.py`. Each pin names a file,
an anchor string that must occur in it, and the claim it supports. The output
(evidence/source_pins.json) stores the file SHA-256, the 1-based line numbers of
every anchor occurrence and the anchor line text. A missing anchor is an error,
not a silent skip. This is source inspection only: nothing here executes the
inspected code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

U = common.UPSTREAM
LR033 = Path("/tmp/kinema-phase2-v2/lib/python3.11/site-packages/lerobot")
LR040 = Path("/tmp/kinema-phase2-v3/lib/python3.11/site-packages/lerobot")
ENV = Path("/home/vx/.cache/kinema-normverify")
LR_OPENPI = ENV / "env-openpi/lib/python3.11/site-packages/lerobot"
LR_FORK = ENV / "env-openpi-vlabench/lib/python3.11/site-packages/lerobot"
KINEMA_ENGINE_COMMIT = "02fbc570034fec2b508ee2682e1f78b5b416bdaf"

PINS = [
    # kinema detector at the board engine (f9a1edd38b66); read from git, not imported
    ("kinema.deploy.stats_fail", "kinema", "src/kinema/checks/deployment.py", 'if feat not in ctx.stats:',
     "deployment-readiness FAILs when the canonical feature has no entry in ctx.stats"),
    ("kinema.deploy.message", "kinema", "src/kinema/checks/deployment.py", 'the policy cannot normalize inputs/outputs at inference',
     "the FAIL text is fixed and policy-agnostic"),
    ("kinema.alias.state", "kinema", "src/kinema/dataset.py", 'STATE_KEY: ("observation.state", "state",',
     "kinema binds a column named `state` to the canonical label observation.state"),
    ("kinema.alias.action", "kinema", "src/kinema/dataset.py", 'ACTION_KEY: ("action", "actions",',
     "kinema binds a column named `actions` to the canonical label action"),
    ("kinema.nan.std_from_loaded_rows", "kinema", "src/kinema/checks/actions.py", "std = np.nanstd(np.where(np.isfinite(arr), arr, np.nan), axis=0)",
     "nan-and-normalization computes std from the loaded (possibly sampled) rows, not from meta/stats.json"),
    ("kinema.nan.tiny_rule", "kinema", "src/kinema/checks/actions.py", "tiny = np.nonzero((std > 0) & (std < 1e-6 * ref_scale))[0]",
     "near-zero rule: 0 < std < 1e-6 * max(|mean|, median live std)"),
    # openpi upstream
    ("openpi.quantile_selection", "openpi", "src/openpi/training/config.py", "use_quantile_norm=model_config.model_type != ModelType.PI0,",
     "upstream openpi: PI0 uses z-score, every other model type (PI05, PI0_FAST) uses quantiles"),
    ("openpi.libero.repack", "openpi", "src/openpi/training/config.py", '"observation/state": "state",',
     "LeRobotLiberoDataConfig repacks dataset keys state/actions/image/wrist_image"),
    ("openpi.load_norm_stats_from_assets", "openpi", "src/openpi/training/config.py", "norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),",
     "training data config reads norm stats from the config assets dir, not from the dataset"),
    ("openpi.train_guard", "openpi", "src/openpi/training/data_loader.py", "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.",
     "training refuses to run without assets norm stats"),
    ("openpi.compute_norm_stats.keys", "openpi", "scripts/compute_norm_stats.py", 'keys = ["state", "actions"]',
     "compute_norm_stats accumulates RunningStats over transformed state/actions from the data loader"),
    ("openpi.compute_norm_stats.save", "openpi", "scripts/compute_norm_stats.py", "output_path = config.assets_dirs / data_config.repo_id",
     "computed stats are written to the config assets dir"),
    ("openpi.checkpoint_save_assets", "openpi", "src/openpi/training/checkpoints.py", "_normalize.save(directory / data_config.asset_id, norm_stats)",
     "training checkpoints copy norm stats into assets/<asset_id>"),
    ("openpi.inference_stats_from_checkpoint", "openpi", "src/openpi/policies/policy_config.py", 'norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)',
     "inference loads norm stats from the checkpoint assets, not the dataset"),
    ("openpi.normalize.none_passthrough", "openpi", "src/openpi/transforms.py", "if self.norm_stats is None:",
     "Normalize with norm_stats None returns data unchanged"),
    ("openpi.normalize.zscore", "openpi", "src/openpi/transforms.py", "return (x - mean) / (std + 1e-6)",
     "z-score epsilon 1e-6"),
    ("openpi.normalize.quantile", "openpi", "src/openpi/transforms.py", "return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0",
     "quantile normalization epsilon 1e-6"),
    ("openpi.create_torch_dataset", "openpi", "src/openpi/training/data_loader.py", "dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)",
     "training data path constructs LeRobotDatasetMetadata/LeRobotDataset from repo_id"),
    # VLABench openpi fork
    ("fork.base_config_no_quantile", "openpi-vlabench", "src/openpi/training/config.py", "def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:",
     "fork base config does not set use_quantile_norm (DataConfig default False)"),
    ("fork.dataconfig_default", "openpi-vlabench", "src/openpi/training/config.py", "use_quantile_norm: bool = False",
     "fork DataConfig default is z-score"),
    ("fork.simple_config_quantile", "openpi-vlabench", "src/openpi/training/config.py", "use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,",
     "only SimpleDataConfig selects quantiles by model type in the fork"),
    ("fork.vlabench.repack", "openpi-vlabench", "src/openpi/training/config.py", '"observation/second_image": "second_image",',
     "LeRobotVLABenchDataConfig repacks image/second_image/wrist_image/state/actions"),
    ("fork.q99_variant", "openpi-vlabench", "src/openpi/training/config.py", "use_quantile_norm=True",
     "one fork config (pifast_ft_vlabench_primitive_aligned_q99) opts into quantiles explicitly"),
    ("fork.train_guard", "openpi-vlabench", "src/openpi/training/data_loader.py", "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.",
     "fork training refuses to run without assets norm stats"),
    ("fork.inference_stats_from_checkpoint", "openpi-vlabench", "src/openpi/policies/policy_config.py", 'norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)',
     "fork inference loads norm stats from checkpoint assets"),
    ("vlabench.uses_fork", "VLABench", "docs/issues.md", "The openpi codebase VLABench used is [openpi](https://github.com/Shiduo-zh/openpi).",
     "VLABench documents its openpi fork"),
    # openpi-pinned lerobot (training data path)
    ("openpi_lerobot.v20_stats_none", "lerobot_openpi", "common/datasets/utils.py", "def load_stats(local_dir: Path)",
     "openpi-pinned lerobot load_stats"),
    ("fork_lerobot.meta_init_stats", "lerobot_fork", "common/datasets/lerobot_dataset.py", "self.stats = load_stats(self.root)",
     "fork-pinned lerobot loads meta/stats.json at metadata construction"),
    # LeRobot 0.3.3
    ("lr033.load_stats_none", "lerobot033", "datasets/utils.py", "if not (local_dir / STATS_PATH).exists():",
     "0.3.3 load_stats returns None when meta/stats.json is absent"),
    ("lr033.v20_branch", "lerobot033", "datasets/lerobot_dataset.py", "self.stats = load_stats(self.root)",
     "v2.0 metadata takes stats from meta/stats.json"),
    ("lr033.v21_branch", "lerobot033", "datasets/lerobot_dataset.py", "self.episodes_stats = load_episodes_stats(self.root)",
     "v2.1 metadata takes stats from meta/episodes_stats.jsonl"),
    ("lr033.feature_state", "lerobot033", "datasets/utils.py", 'elif key.startswith("observation"):',
     "only observation* keys become STATE features"),
    ("lr033.feature_action", "lerobot033", "datasets/utils.py", 'elif key.startswith("action"):',
     "keys starting with action (including actions) become ACTION features"),
    ("lr033.make_policy_stats", "lerobot033", "policies/factory.py", 'kwargs["dataset_stats"] = ds_meta.stats',
     "make_policy passes dataset metadata stats to the policy normalizers"),
    ("lr033.inf_init", "lerobot033", "policies/normalize.py", "mean = torch.ones(shape, dtype=torch.float32) * torch.inf",
     "normalizer buffers start at infinity when no stats are given"),
    ("lr033.inf_assert", "lerobot033", "policies/normalize.py", 'assert not torch.isinf(mean).any(), _no_stats_error_str("mean")',
     "normalization asserts the buffers were overwritten"),
    ("lr033.meanstd_eps", "lerobot033", "policies/normalize.py", "batch[key] = (batch[key] - mean) / (std + 1e-8)",
     "0.3.3 MEAN_STD epsilon 1e-8"),
    ("lr033.minmax_eps", "lerobot033", "policies/normalize.py", "batch[key] = (batch[key] - min) / (max - min + 1e-8)",
     "0.3.3 MIN_MAX epsilon 1e-8"),
    # LeRobot 0.4.0
    ("lr040.train_stats", "lerobot040", "scripts/lerobot_train.py", 'processor_kwargs["dataset_stats"] = dataset.meta.stats',
     "lerobot-train builds the normalizer from dataset.meta.stats"),
    ("lr040.missing_passthrough", "lerobot040", "processor/normalize_processor.py", "if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:",
     "a feature without stats is returned unchanged"),
    ("lr040.meanstd", "lerobot040", "processor/normalize_processor.py", "denom = std + self.eps",
     "MEAN_STD adds eps to std"),
    ("lr040.minmax_zero_only", "lerobot040", "processor/normalize_processor.py", "return 2 * (tensor - min_val) / denom - 1",
     "MIN_MAX substitutes eps only when max == min exactly"),
    ("lr040.quantile_zero_only", "lerobot040", "processor/normalize_processor.py", "return 2.0 * (tensor - q01) / denom - 1.0",
     "QUANTILES substitutes eps only when q99 == q01 exactly"),
    ("lr040.pi05_quantiles", "lerobot040", "policies/pi05/configuration_pi05.py", '"STATE": NormalizationMode.QUANTILES,',
     "LeRobot pi05 default STATE normalization is QUANTILES"),
    ("lr040.pi0_meanstd", "lerobot040", "policies/pi0/configuration_pi0.py", '"STATE": NormalizationMode.MEAN_STD,',
     "LeRobot pi0 default STATE normalization is MEAN_STD"),
    # Isaac-GR00T n1.6.1
    ("gr00t.factory_generate_stats", "gr00t", "gr00t/data/dataset/factory.py", "generate_stats(dataset_path)",
     "training generates stats before building each dataset"),
    ("gr00t.generate_stats_features", "gr00t", "gr00t/data/stats.py", 'le_features = json.load(f)["features"]',
     "stats generation reads info.json features"),
    ("gr00t.loader_asserts_stats", "gr00t", "gr00t/data/dataset/lerobot_episode_loader.py", "please use gr00t/data/stats.py to generate it",
     "the episode loader asserts meta/stats.json exists"),
    ("gr00t.behavior_state_keys", "gr00t", "gr00t/configs/data/embodiment_configs.py", '"behavior_r1_pro": {',
     "behavior_r1_pro modality config (state keys listed below this line)"),
    ("gr00t.minmax_isclose", "gr00t", "gr00t/data/utils.py", "mask = ~np.isclose(max_vals, min_vals)",
     "min/max normalization leaves channels with isclose(max, min) at 0"),
    ("gr00t.minmax_clip", "gr00t", "gr00t/data/state_action/state_action_processor.py", "normalized = np.clip(normalized, -1.0, 1.0)",
     "with clip_outliers the min/max output is clipped to [-1, 1]"),
]

ROOTS = {
    "openpi": U / "openpi",
    "openpi-vlabench": U / "openpi-vlabench",
    "gr00t": U / "Isaac-GR00T-n1.6.1",
    "VLABench": U / "VLABench",
    "lerobot033": LR033,
    "lerobot040": LR040,
    "lerobot_openpi": LR_OPENPI,
    "lerobot_fork": LR_FORK,
}


def read_source(root: str, rel: str) -> bytes:
    if root == "kinema":
        return subprocess.run(["git", "-C", str(common.REPO), "show", f"{KINEMA_ENGINE_COMMIT}:{rel}"],
                              check=True, capture_output=True).stdout
    return (ROOTS[root] / rel).read_bytes()


def main() -> None:
    out = []
    for pin_id, root, rel, anchor, claim in PINS:
        data = read_source(root, rel)
        lines = data.decode().splitlines()
        hits = [i + 1 for i, line in enumerate(lines) if anchor in line]
        if not hits:
            raise SystemExit(f"anchor not found for {pin_id}: {anchor!r} in {root}:{rel}")
        out.append({"id": pin_id, "root": root, "path": rel, "sha256": common.sha256_bytes(data),
                    "lines": hits, "anchor": anchor, "line_text": lines[hits[0] - 1].strip(), "claim": claim})
    common.write_json(common.EVIDENCE / "source_pins.json", {
        "schema": "kinema.normverify.source_pins/1",
        "kinema_engine_commit": KINEMA_ENGINE_COMMIT,
        "roots": {k: str(v) for k, v in ROOTS.items()},
        "pins": out,
    })
    print(f"{len(out)} pins")


if __name__ == "__main__":
    main()
