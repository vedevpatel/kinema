"""RLDS checks: metadata-integrity audits for TFDS/TFRecord datasets on GCS.

These run on an RldsDataset (a probed GCS prefix), not a DatasetContext, so
they are not part of ALL_CHECKS and never run in the LeRobot audit path -
the leaderboard's rlds lane calls them directly. They reuse the shared
Finding/CheckResult/Severity framework. Only rlds_step_structure touches the
network (sample mode, off by default): the rest read the probe's listing and
the two small JSON metadata files.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from kinema.checks.base import CheckResult, Finding, Severity
from kinema.rlds import RldsDataset, RldsError, GcsClient, decode_example, iter_records

SHARD_RE = re.compile(
    r"-(?P<split>[A-Za-z0-9_]+)\.(?:tfrecord|array_record)-(?P<idx>\d{5})-of-(?P<total>\d{5})$")
SINGLE_SHARD_WARN_BYTES = 2 * 1024**3  # one corrupt object kills everything
DEFAULT_SAMPLE_CAP = 64 * 1024**2  # skip episodes whose record is bigger


@dataclass
class RldsCheckConfig:
    sample: bool = False  # fetch and decode one episode (network)
    sample_cap_bytes: int = DEFAULT_SAMPLE_CAP


def _shard_parts(ds: RldsDataset) -> dict[str, dict[int, int]]:
    """split -> {shard index: size}; unparseable names land under split ''."""
    out: dict[str, dict[int, int]] = {}
    for name, size in ds.shards:
        base = name.rsplit("/", 1)[-1]
        m = SHARD_RE.search(base)
        if not m:
            out.setdefault("", {})[-1] = size
            continue
        out.setdefault(m.group("split"), {})[int(m.group("idx"))] = size
    return out


def rlds_episode_counts(ds: RldsDataset) -> CheckResult:
    """Declared shard/episode counts vs what the bucket actually holds."""
    res = CheckResult(check_id="rlds-episode-counts", title="RLDS episode counts")
    shards = _shard_parts(ds)
    declared = ds.shard_lengths
    if not declared:
        res.findings.append(Finding(
            "rlds-episode-counts", Severity.WARN,
            "dataset_info.json declares no splits with shardLengths"))
    for split, lengths in declared.items():
        actual = shards.get(split, {})
        if len(actual) != len(lengths):
            res.findings.append(Finding(
                "rlds-episode-counts", Severity.FAIL,
                f"split {split!r}: dataset_info declares {len(lengths)} shards, "
                f"bucket holds {len(actual)}",
                details={"split": split, "declared": len(lengths), "actual": len(actual)}))
        if not actual and lengths:
            continue
        # shardLengths entries of 0 would make a shard undeletable-quiet
        zero = [i for i, n in enumerate(lengths) if n == 0]
        if zero:
            res.findings.append(Finding(
                "rlds-episode-counts", Severity.WARN,
                f"split {split!r}: shards {zero} declared with 0 episodes",
                details={"split": split, "shards": zero}))
    for split in shards:
        if split and split not in declared:
            res.findings.append(Finding(
                "rlds-episode-counts", Severity.FAIL,
                f"split {split!r} has shard files but no dataset_info entry",
                details={"split": split}))
    total = sum(ds.splits.values())
    res.summary = (f"{total} episodes across {len(declared)} split(s)"
                   if declared else "no declared splits")
    return res


def rlds_feature_spec(ds: RldsDataset) -> CheckResult:
    """features.json sanity: steps sequence, action/observation, image features."""
    res = CheckResult(check_id="rlds-feature-spec", title="RLDS feature spec")
    tree = ds.features.get("featuresDict", {}).get("features", {})
    if not tree:
        res.findings.append(Finding(
            "rlds-feature-spec", Severity.FAIL,
            "features.json has no featuresDict.features tree"))
        res.summary = "feature tree missing"
        return res
    steps = tree.get("steps", {})
    step_feats = (steps.get("sequence", {}).get("feature", {})
                  .get("featuresDict", {}).get("features", {}))
    if not step_feats:
        res.findings.append(Finding(
            "rlds-feature-spec", Severity.FAIL,
            "features.json has no steps sequence feature"))
        res.summary = "no steps sequence"
        return res
    for required in ("is_first", "is_last"):
        if required not in step_feats:
            res.findings.append(Finding(
                "rlds-feature-spec", Severity.FAIL,
                f"steps feature missing {required!r}"))
    if "action" not in step_feats:
        res.findings.append(Finding(
            "rlds-feature-spec", Severity.WARN, "steps feature has no action"))
    obs = step_feats.get("observation", {})
    obs_feats = obs.get("featuresDict", {}).get("features", {})
    if not obs_feats:
        res.findings.append(Finding(
            "rlds-feature-spec", Severity.WARN, "steps feature has no observation"))

    def _images(node: dict) -> int:
        n = 0
        cls = node.get("pythonClassName", "")
        if "Image" in cls:
            n += 1
        for child in node.get("featuresDict", {}).get("features", {}).values():
            n += _images(child)
        seq = node.get("sequence", {}).get("feature")
        if seq:
            n += _images(seq)
        return n

    images = _images(steps)
    if images == 0:
        res.findings.append(Finding(
            "rlds-feature-spec", Severity.INFO, "no image features declared"))
    res.summary = (f"{len(step_feats)} step features, {images} image feature(s), "
                   f"{len(obs_feats)} observation feature(s)")
    return res


def rlds_shard_inventory(ds: RldsDataset) -> CheckResult:
    """Shard files: parseable names, contiguous numbering, non-zero, sized."""
    res = CheckResult(check_id="rlds-shard-inventory", title="RLDS shard inventory")
    shards = _shard_parts(ds)
    if "" in shards:
        res.findings.append(Finding(
            "rlds-shard-inventory", Severity.WARN,
            f"{len(shards[''])} shard file(s) with unparseable names"))
    for split, idx_map in shards.items():
        if split == "":
            continue
        declared_total = None
        for name, _ in ds.shards:
            m = SHARD_RE.search(name.rsplit("/", 1)[-1])
            if m and m.group("split") == split:
                declared_total = int(m.group("total"))
                break
        idxs = sorted(idx_map)
        missing = [i for i in range(idxs[-1] + 1) if i not in idx_map] if idxs else []
        if missing:
            res.findings.append(Finding(
                "rlds-shard-inventory", Severity.FAIL,
                f"split {split!r}: missing shard(s) {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}",
                details={"split": split, "missing": missing}))
        if declared_total is not None and declared_total != len(idxs):
            res.findings.append(Finding(
                "rlds-shard-inventory", Severity.FAIL,
                f"split {split!r}: filenames declare {declared_total} shards, "
                f"bucket holds {len(idxs)}",
                details={"split": split, "declared": declared_total, "actual": len(idxs)}))
        zero = [i for i, s in idx_map.items() if s == 0]
        if zero:
            res.findings.append(Finding(
                "rlds-shard-inventory", Severity.FAIL,
                f"split {split!r}: zero-byte shard(s) {zero}",
                details={"split": split, "shards": zero}))
        if len(idxs) == 1 and idx_map[idxs[0]] > SINGLE_SHARD_WARN_BYTES:
            res.findings.append(Finding(
                "rlds-shard-inventory", Severity.WARN,
                f"split {split!r}: single shard of {idx_map[idxs[0]] / 1024**3:.1f} GiB - "
                "one corrupt object kills the whole split",
                details={"split": split, "bytes": idx_map[idxs[0]]}))
    res.summary = f"{sum(len(v) for k, v in shards.items() if k)} shards, {ds.data_bytes / 1024**3:.1f} GiB"
    return res


def rlds_step_structure(ds: RldsDataset, client: GcsClient,
                        cfg: RldsCheckConfig | None = None) -> CheckResult:
    """Sample one episode: framing flags sane, per-step feature lengths agree.

    The episode record's 12-byte header is read first; an episode over the
    sample cap is skipped (INFO) rather than downloaded.
    """
    cfg = cfg or RldsCheckConfig()
    res = CheckResult(check_id="rlds-step-structure", title="RLDS step structure")
    if not cfg.sample:
        res.skipped = True
        res.skip_reason = "sample mode off"
        return res
    if not ds.shards:
        res.skipped = True
        res.skip_reason = "no shards"
        return res
    name = sorted(ds.shards)[0][0]
    if ".tfrecord-" not in name:
        # ArrayRecord shards (bc_z): the payload framing is not TFRecord, so
        # the sample decode would read garbage - skip it; the metadata checks
        # still ran
        res.skipped = True
        res.skip_reason = "ArrayRecord shards: payload sampling not supported"
        return res
    try:
        head = client.get(name, (0, 11))
        (length,) = struct.unpack_from("<Q", head, 0)
        if length > cfg.sample_cap_bytes:
            res.skipped = True
            res.skip_reason = (f"first episode is {length / 1024**2:.0f} MiB, "
                               f"over the {cfg.sample_cap_bytes / 1024**2:.0f} MiB sample cap")
            return res
        buf = client.get(name, (0, 12 + length + 3))
        record = next(iter_records(buf), None)
        if record is None:
            res.findings.append(Finding(
                "rlds-step-structure", Severity.FAIL,
                "first shard's first record did not decode"))
            return res
        ep = decode_example(record)
    except RldsError as e:
        res.findings.append(Finding(
            "rlds-step-structure", Severity.FAIL, f"sample decode failed: {e}"))
        return res
    is_first = ep.get("steps/is_first", {}).get("int64", [])
    is_last = ep.get("steps/is_last", {}).get("int64", [])
    n = len(is_first)
    if not n:
        res.findings.append(Finding(
            "rlds-step-structure", Severity.FAIL, "episode has no steps"))
        return res
    if is_first[0] != 1 or sum(is_first) != 1:
        res.findings.append(Finding(
            "rlds-step-structure", Severity.FAIL,
            f"is_first broken: first={is_first[0]}, sum={sum(is_first)} over {n} steps"))
    if len(is_last) != n or is_last[-1] != 1 or sum(is_last) != 1:
        res.findings.append(Finding(
            "rlds-step-structure", Severity.FAIL,
            "is_last broken over the episode"))
    for key, feat in ep.items():
        if not key.startswith("steps/"):
            continue
        kind = next(iter(feat))
        count = len(feat[kind])
        if count % n:
            res.findings.append(Finding(
                "rlds-step-structure", Severity.WARN,
                f"{key}: {count} values not a multiple of {n} steps",
                details={"feature": key, "count": count, "steps": n}))
    res.summary = f"sampled episode: {n} steps, {len(ep)} features"
    return res


def run_rlds_checks(ds: RldsDataset, client: GcsClient | None = None,
                    cfg: RldsCheckConfig | None = None,
                    mirror: RldsDataset | None = None,
                    mirror_bucket: str = "") -> list[CheckResult]:
    """The metadata-only set, plus the sampled check when a client is given,
    plus the mirror-drift check when a second-bucket copy was probed."""
    out = [rlds_episode_counts(ds), rlds_feature_spec(ds), rlds_shard_inventory(ds)]
    out.append(rlds_step_structure(ds, client or GcsClient(""), cfg))
    if mirror is not None:
        out.append(rlds_mirror_drift(ds, mirror, mirror_bucket))
    return out


MIRROR_BYTES_WARN = 0.05  # copies drifting further than this look stale, not just rebuilt


def rlds_mirror_drift(ds: RldsDataset, mirror: RldsDataset, mirror_bucket: str) -> CheckResult:
    """Same dataset in a second bucket: do the two copies agree?

    The corpus lives in both gdm-robotics-open-x-embodiment and
    gresearch/robotics, and the copies DIVERGE (agent_aware_affordances: 63
    shards / 18.1 GB in gdm vs 64 / 18.5 GB in gresearch at the same
    version). A row audits one bucket's copy; this check surfaces the other
    copy's disagreement instead of leaving it silent. Transparency only: it
    never FAILs an audit - INFO when the copies drift at the same version
    and episode count, WARN when they disagree on version or episodes or
    drift far in bytes.
    """
    res = CheckResult(check_id="rlds-mirror-drift",
                      title="RLDS mirror drift (second-bucket copy)")
    my_bytes, their_bytes = ds.data_bytes, mirror.data_bytes
    drift = abs(my_bytes - their_bytes) / max(my_bytes, their_bytes, 1)
    if (ds.version == mirror.version and ds.episodes == mirror.episodes
            and len(ds.shards) == len(mirror.shards) and drift == 0):
        return res
    sev = (Severity.WARN
           if (ds.version != mirror.version or ds.episodes != mirror.episodes
               or drift > MIRROR_BYTES_WARN)
           else Severity.INFO)
    res.findings.append(Finding(
        "rlds-mirror-drift", sev,
        f"{mirror_bucket} copy diverges: v{mirror.version} vs v{ds.version}, "
        f"{mirror.episodes} vs {ds.episodes} episodes, {len(mirror.shards)} vs "
        f"{len(ds.shards)} shards, {their_bytes / 1024**3:.1f} vs "
        f"{my_bytes / 1024**3:.1f} GB ({drift:.1%} apart)",
        details={"mirror_bucket": mirror_bucket, "version": ds.version,
                 "mirror_version": mirror.version, "episodes": ds.episodes,
                 "mirror_episodes": mirror.episodes, "shards": len(ds.shards),
                 "mirror_shards": len(mirror.shards), "bytes": my_bytes,
                 "mirror_bytes": their_bytes}))
    return res
