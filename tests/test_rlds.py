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
    RldsCheckConfig, rlds_episode_counts, rlds_feature_spec, rlds_shard_inventory,
    rlds_step_structure, run_rlds_checks,
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
    extra = _dataset(shards=_dataset().shards
                     + [("prefix/austin_buds_dataset_converted_externally_to_rlds"
                         "-train.tfrecord-00001-of-00002", 100)])
    res = rlds_episode_counts(extra)
    assert res.verdict == Severity.FAIL
    assert any("declares 1 shards" in f.message for f in res.findings)
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
