"""Tests for the RLDS reader (kinema.rlds) and the RLDS check family."""

import json
import struct
from pathlib import Path

import pytest

from kinema.rlds import (
    GcsClient, RldsDataset, RldsError, crc32c, decode_example, iter_records,
    _masked_crc32c,
)
from kinema.checks.base import Severity
from kinema.checks.rlds_checks import (
    RldsCheckConfig, rlds_episode_counts, rlds_feature_spec, rlds_mirror_drift,
    rlds_shard_inventory, rlds_step_structure, run_rlds_checks,
)

FIX = Path(__file__).parent / "fixtures" / "rlds" / "austin_buds" / "0.1.0"
SHARD = FIX / ("austin_buds_dataset_converted_externally_to_rlds"
               "-train.tfrecord-00000-of-00001")


def _record(payload: bytes) -> bytes:
    head = struct.pack("<Q", len(payload))
    return head + struct.pack("<I", _masked_crc32c(head)) + payload + struct.pack(
        "<I", _masked_crc32c(payload))


def _dataset(**overrides) -> RldsDataset:
    kw = dict(
        name="austin_buds_dataset_converted_externally_to_rlds",
        version="0.1.0",
        info=json.loads((FIX / "dataset_info.json").read_text()),
        features=json.loads((FIX / "features.json").read_text()),
        shards=[(f"prefix/{SHARD.name}", SHARD.stat().st_size)],
    )
    kw.update(overrides)
    return RldsDataset(**kw)


# --- CRC / framing ----------------------------------------------------------

def test_crc32c_check_value():
    # the canonical CRC32C check vector
    assert crc32c(b"123456789") == 0xE3069283


def test_iter_records_round_trip_and_clean_tail():
    buf = _record(b"hello") + _record(b"world")
    assert list(iter_records(buf)) == [b"hello", b"world"]
    # a range-fetched buffer ending mid-record stops cleanly
    assert list(iter_records(buf + _record(b"truncated...")[:10])) == [b"hello", b"world"]


def test_iter_records_rejects_corruption():
    good = _record(b"payload")
    bad_len = bytearray(good)
    bad_len[8] ^= 0xFF  # break the length CRC
    with pytest.raises(RldsError):
        list(iter_records(bytes(bad_len)))
    bad_payload = bytearray(good)
    bad_payload[14] ^= 0xFF  # break the payload CRC
    with pytest.raises(RldsError):
        list(iter_records(bytes(bad_payload)))


# --- tf.Example decode on real payloads --------------------------------------

def test_decode_real_episode_fixture():
    ep = decode_example(next(iter_records(SHARD.read_bytes())))
    assert ep["steps/is_first"]["int64"] == [1, 0, 0]
    assert ep["steps/is_last"]["int64"] == [0, 0, 1]
    assert len(ep["steps/action"]["float"]) == 21  # 3 steps x 7-dim
    assert len(ep["steps/reward"]["float"]) == 3
    # real PNG frames, straight from TFDS output
    assert ep["steps/observation/image"]["bytes"][0][:4].hex() == "89504e47"
    assert len(ep["steps/observation/image"]["bytes"]) == 3
    assert ep["steps/language_instruction"]["bytes"][0].startswith(b"Take the lid")
    assert ep["episode_metadata/file_path"]["bytes"]


# --- RldsDataset ---------------------------------------------------------------

def test_dataset_splits_from_shard_lengths():
    ds = _dataset()
    assert ds.splits == {"train": 1}
    assert ds.shard_lengths == {"train": [1]}
    assert ds.episodes == 1
    assert ds.data_bytes == SHARD.stat().st_size


class _StubClient:
    def __init__(self, files):
        self.files = files  # name -> bytes
        self.calls = 0

    def list(self, prefix="", delimiter=""):
        self.calls += 1
        prefixes = sorted({n[len(prefix):].split("/")[0] + "/"
                           for n in self.files
                           if n.startswith(prefix) and "/" in n[len(prefix):]})
        items = [(n, len(b)) for n, b in self.files.items()
                 if n.startswith(prefix) and "/" not in n[len(prefix):]]
        return ([prefix + p for p in prefixes] if delimiter else [], items)

    def get(self, name, byte_range=None):
        self.calls += 1
        b = self.files[name]
        if byte_range:
            return b[byte_range[0]:byte_range[1] + 1]
        return b

    def get_json(self, name):
        return json.loads(self.get(name))


def test_probe_picks_newest_semver_and_filters_shards():
    base = "ds/"
    files = {
        "ds/0.1.0/dataset_info.json": b'{"splits": []}',
        "ds/0.1.0/features.json": b"{}",
        "ds/0.1.0/ds-train.tfrecord-00000-of-00001": b"old",
        "ds/0.2.0/dataset_info.json": b'{"splits": [{"name": "train", "shardLengths": ["2"]}]}',
        "ds/0.2.0/features.json": b"{}",
        "ds/0.2.0/ds-train.tfrecord-00000-of-00001": b"new!",
        "ds/0.2.0/README.md": b"not a shard",
    }
    ds = RldsDataset.probe(_StubClient(files), base)
    assert ds.version == "0.2.0"
    assert [n.rsplit("/", 1)[-1] for n, _ in ds.shards] == ["ds-train.tfrecord-00000-of-00001"]
    assert ds.splits == {"train": 2}


def test_probe_reads_folder_marker_objects():
    # GCS has no directories: some datasets expose version dirs ONLY as
    # <version>_$folder$ placeholder objects (bc_z's shape)
    files = {
        "ds/0.1.0_$folder$": b"x",
        "ds/1.0.1_$folder$": b"x",
        "ds/old1.0.1_$folder$": b"x",  # not semver: never picked
        "ds/1.0.1/dataset_info.json": b'{"splits": []}',
        "ds/1.0.1/features.json": b"{}",
        "ds/1.0.1/ds-train.tfrecord-00000-of-00001": b"data",
    }
    ds = RldsDataset.probe(_StubClient(files), "ds/")
    assert ds.version == "1.0.1"
    assert len(ds.shards) == 1


def test_probe_skips_placeholder_only_prefixes_fast():
    # 44 of 73 Open X prefixes are markers with no data: skip without
    # fetching metadata that does not exist
    client = _StubClient({"ds/0.1.0_$folder$": b"x"})
    with pytest.raises(RldsError, match="no dataset_info.json"):
        RldsDataset.probe(client, "ds/")
    assert client.calls <= 2  # one prefix listing + one version listing


# --- checks ---------------------------------------------------------------------

def test_episode_counts_ok_and_mismatch():
    assert rlds_episode_counts(_dataset()).verdict == Severity.OK
    extra = _dataset(info={"splits": [{"name": "train", "shardLengths": ["1", "1"]}]},
                     shards=_dataset().shards
                     + [("prefix/austin_buds_dataset_converted_externally_to_rlds"
                         "-train.tfrecord-00001-of-00002", 100)])
    res = rlds_episode_counts(extra)
    assert res.verdict == Severity.FAIL
    assert any("declares 2 shards" in f.message for f in res.findings)
    noinfo = _dataset(info={"splits": []}, shards=[])
    assert rlds_episode_counts(noinfo).verdict == Severity.WARN
    # shards on the bucket with no dataset_info entry is a FAIL, not a WARN
    orphan = _dataset(info={"splits": []})
    assert rlds_episode_counts(orphan).verdict == Severity.FAIL


def test_feature_spec_ok_and_missing_pieces():
    assert rlds_feature_spec(_dataset()).verdict == Severity.OK
    res = rlds_feature_spec(_dataset(features={}))
    assert res.verdict == Severity.FAIL
    bare = _dataset(features={"featuresDict": {"features": {"steps": {
        "sequence": {"feature": {"featuresDict": {"features": {
            "is_first": {}, "is_last": {}}}}}}}}})
    res2 = rlds_feature_spec(bare)
    assert res2.verdict == Severity.WARN  # no action, no observation
    assert any("no action" in f.message for f in res2.findings)


def test_shard_inventory_defects():
    assert rlds_shard_inventory(_dataset()).verdict == Severity.OK
    zero = _dataset(shards=[(_dataset().shards[0][0], 0)])
    assert rlds_shard_inventory(zero).verdict == Severity.FAIL
    gap = _dataset(shards=[
        ("prefix/x-train.tfrecord-00000-of-00003", 10),
        ("prefix/x-train.tfrecord-00002-of-00003", 10)])
    res = rlds_shard_inventory(gap)
    assert res.verdict == Severity.FAIL
    assert any("missing shard" in f.message for f in res.findings)
    huge = _dataset(shards=[(_dataset().shards[0][0], 3 * 1024**3)])
    assert rlds_shard_inventory(huge).verdict == Severity.WARN


def test_step_structure_sample_modes():
    ds = _dataset()
    off = rlds_step_structure(ds, GcsClient("unused"), RldsCheckConfig(sample=False))
    assert off.skipped
    client = _StubClient({ds.shards[0][0]: SHARD.read_bytes()})
    on = rlds_step_structure(ds, client, RldsCheckConfig(sample=True))
    assert not on.skipped and on.verdict == Severity.OK, on.findings
    assert "3 steps" in on.summary
    capped = rlds_step_structure(ds, client, RldsCheckConfig(sample=True, sample_cap_bytes=10))
    assert capped.skipped and "sample cap" in capped.skip_reason
    # valid framing, corrupted payload CRC -> decode failure is a FAIL
    corrupt = bytearray(_record(b"payload"))
    corrupt[14] ^= 0xFF
    bad_client = _StubClient({ds.shards[0][0]: bytes(corrupt)})
    bad = rlds_step_structure(ds, bad_client, RldsCheckConfig(sample=True))
    assert bad.verdict == Severity.FAIL


def test_run_rlds_checks_metadata_only():
    results = run_rlds_checks(_dataset())
    assert [r.check_id for r in results] == [
        "rlds-episode-counts", "rlds-feature-spec", "rlds-shard-inventory",
        "rlds-step-structure"]
    assert [r.verdict for r in results[:3]] == [Severity.OK] * 3
    assert results[3].skipped


def test_array_record_shards_inventory_parse_and_sample_skip():
    # bc_z on gresearch stores shards as ArrayRecord, not TFRecord
    files = {
        "ds/1.0.1/dataset_info.json": b'{"splits": [{"name": "train", "shardLengths": ["5", "5"]}]}',
        "ds/1.0.1/features.json": b"{}",
        "ds/1.0.1/ds-train.array_record-00000-of-00002": b"a" * 100,
        "ds/1.0.1/ds-train.array_record-00001-of-00002": b"b" * 100,
    }
    ds = RldsDataset.probe(_StubClient(files), "ds/")
    assert len(ds.shards) == 2 and ds.data_bytes == 200  # inventoried + counted
    assert rlds_shard_inventory(ds).verdict == Severity.OK  # SHARD_RE parses them
    assert rlds_episode_counts(ds).verdict == Severity.OK
    res = rlds_step_structure(ds, _StubClient(files), RldsCheckConfig(sample=True))
    assert res.skipped and "ArrayRecord" in res.skip_reason  # no garbage decode


def test_mirror_drift_agree_info_warn():
    ds = _dataset()
    # identical copies: silent
    same = _dataset()
    res = rlds_mirror_drift(ds, same, "gresearch")
    assert not res.findings and res.verdict == Severity.OK
    # agent_aware_affordances-shaped: one extra shard, ~2% bytes, same
    # version and episodes -> INFO (transparency, not an alarm)
    drifted = _dataset(shards=ds.shards + [(ds.shards[0][0].replace("00000-of", "00001-of"),
                                            int(ds.shards[0][1] * 0.02))])
    res = rlds_mirror_drift(ds, drifted, "gresearch")
    assert res.verdict == Severity.INFO and res.findings[0].details["mirror_shards"] == 2
    # version mismatch -> WARN
    res = rlds_mirror_drift(ds, _dataset(version="9.9.9"), "gresearch")
    assert res.verdict == Severity.WARN
    # episode-count mismatch -> WARN
    other_info = {"splits": [{"name": "train", "shardLengths": ["999"]}]}
    res = rlds_mirror_drift(ds, _dataset(info=other_info), "gresearch")
    assert res.verdict == Severity.WARN
    # same version and episodes but bytes drifted past 5% -> WARN
    fat = _dataset(shards=[(ds.shards[0][0], ds.shards[0][1] * 2)])
    res = rlds_mirror_drift(ds, fat, "gresearch")
    assert res.verdict == Severity.WARN
    # never FAILs an audit: transparency only
    assert all(r.verdict <= Severity.WARN for r in
               [rlds_mirror_drift(ds, m, "gresearch") for m in (same, drifted, fat)])
    # run_rlds_checks appends the drift check only when a mirror was probed
    with_mirror = run_rlds_checks(ds, mirror=same, mirror_bucket="gresearch")
    assert [r.check_id for r in with_mirror][-1] == "rlds-mirror-drift"
    assert [r.check_id for r in run_rlds_checks(ds)][-1] != "rlds-mirror-drift"


def test_unsuffixed_shard_names_are_counted_and_gaps_still_fail():
    info = {"splits": [{"name": "train", "shardLengths": ["1", "1", "1"],
                        "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}"}]}
    shards = [(f"prefix/ds-train.tfrecord-{i:05d}", 10) for i in range(3)]
    ds = _dataset(info=info, shards=shards)
    assert rlds_episode_counts(ds).verdict == Severity.OK
    assert rlds_shard_inventory(ds).verdict == Severity.OK
    missing = _dataset(info=info, shards=[shards[0], shards[2]])
    assert rlds_episode_counts(missing).verdict == Severity.FAIL
    assert rlds_shard_inventory(missing).verdict == Severity.FAIL


def test_declared_shard_generation_selected_without_union():
    info = {"splits": [{"name": "train", "shardLengths": ["1", "1"]}]}
    shards = [(f"prefix/ds-train.tfrecord-{i:05d}-of-00002", 10) for i in range(2)]
    shards += [(f"prefix/ds-train.tfrecord-{i:05d}-of-00003", 20) for i in range(3)]
    ds = _dataset(info=info, shards=shards)
    assert len(ds.shards) == 2 and len(ds.ignored_shards) == 3
    assert ds.data_bytes == 20
    assert rlds_episode_counts(ds).verdict == Severity.OK
    assert rlds_shard_inventory(ds).verdict == Severity.OK
    missing = _dataset(info=info, shards=shards[1:])
    assert rlds_episode_counts(missing).verdict == Severity.FAIL


def test_ambiguous_generations_skip_instead_of_unioning():
    info = {"splits": [{"name": "train", "shardLengths": ["1"]}]}
    ds = _dataset(info=info, shards=[("p/a-train.tfrecord-00000-of-00001", 1),
                                    ("p/b-train.tfrecord-00000-of-00001", 1)])
    assert ds.ambiguous_splits == ["train"]
    assert rlds_episode_counts(ds).skipped
    assert rlds_shard_inventory(ds).skipped


def test_legacy_feature_schema_and_nontrajectory_applicability():
    def legacy(node):
        cls = node.get("pythonClassName", "")
        if "featuresDict" in node:
            return {"type": "tf.FeaturesDict", "content": {
                k: legacy(v) for k, v in node["featuresDict"]["features"].items()}}
        if "sequence" in node:
            return {"type": "tf.Dataset", "content": {"feature": legacy(node["sequence"]["feature"])}}
        return {"type": cls, "content": {}}
    modern = _dataset()
    old = _dataset(features=legacy(modern.features))
    assert rlds_feature_spec(old).verdict == Severity.OK
    flat = _dataset(features={"featuresDict": {"features": {"question": {"text": {}}}}})
    assert rlds_feature_spec(flat).skipped
    assert rlds_step_structure(flat, GcsClient("unused"), RldsCheckConfig(sample=True)).skipped
    assert rlds_feature_spec(_dataset(features={})).verdict == Severity.FAIL
