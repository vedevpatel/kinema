"""Build the checked-in RLDS fixture from REAL Open X bytes.

Provenance: austin_buds_dataset_converted_externally_to_rlds 0.1.0, shard 0
(one real 640-step episode) fetched anonymously from
gs://gdm-robotics-open-x-embodiment. This script slices the first 3 steps of
every steps/* feature and re-encodes a mini episode with the real per-step
payloads (PNG frames, action/state floats, instruction strings) untouched.
The container framing + tf.Example encoding here is ours; the payloads are
TFDS-real, which is what grounds the reader. dataset_info.json is adjusted
to stay self-consistent (1 shard, 3 episodes... 3 steps in 1 episode).

Run:  python tests/fixtures/rlds/build_fixture.py /tmp/shard0_full.bin
Not part of the test suite; the fixture bytes are checked in.
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from kinema.rlds import decode_example, iter_records, _masked_crc32c  # noqa: E402

OUT = Path(__file__).parent / "austin_buds" / "0.1.0"
KEEP_STEPS = 3


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint(field << 3 | wire)


def _ld(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def encode_example(features: dict[str, tuple[str, list]]) -> bytes:
    """features: name -> ("bytes"|"int64"|"float", values) -> serialized Example."""
    entries = []
    for name, (kind, values) in sorted(features.items()):
        if kind == "bytes":
            body = b"".join(_ld(1, v) for v in values)
            feature = _ld(1, body)
        elif kind == "int64":
            packed = b"".join(_varint(v % (1 << 64)) for v in values)
            feature = _ld(3, _ld(1, packed))
        elif kind == "float":
            packed = b"".join(struct.pack("<f", v) for v in values)
            feature = _ld(2, _ld(1, packed))
        else:
            raise ValueError(kind)
        entry = _ld(1, name.encode()) + _ld(2, feature)
        entries.append(_ld(1, entry))
    return _ld(1, b"".join(entries))


def write_record(payload: bytes) -> bytes:
    head = struct.pack("<Q", len(payload))
    return (head + struct.pack("<I", _masked_crc32c(head))
            + payload + struct.pack("<I", _masked_crc32c(payload)))


def main(shard_path: str) -> None:
    buf = Path(shard_path).read_bytes()
    ep = decode_example(next(iter_records(buf)))
    n = len(ep["steps/is_first"]["int64"])
    k = KEEP_STEPS
    out: dict[str, tuple[str, list]] = {}
    for key, feat in ep.items():
        kind = next(iter(feat))
        vals = feat[kind]
        if key == "episode_metadata/file_path":
            out[key] = (kind, vals[:1])
        elif key.startswith("steps/"):
            per = len(vals) // n
            out[key] = (kind, vals[:per * k])
    # fix the framing flags for a 3-step episode
    out["steps/is_first"] = ("int64", [1] + [0] * (k - 1))
    out["steps/is_last"] = ("int64", [0] * (k - 1) + [1])
    out["steps/is_terminal"] = ("int64", [0] * (k - 1) + [1])
    record = encode_example(out)
    shard = write_record(record)
    OUT.mkdir(parents=True, exist_ok=True)
    shard_name = ("austin_buds_dataset_converted_externally_to_rlds"
                  "-train.tfrecord-00000-of-00001")
    (OUT / shard_name).write_bytes(shard)
    info = json.loads((OUT / "dataset_info.json").read_text())
    info["splits"] = [{
        "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
        "name": "train",
        "numBytes": str(len(shard)),
        "shardLengths": ["1"],
    }]
    (OUT / "dataset_info.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {OUT / shard_name} ({len(shard):,} bytes), 1 episode x {k} steps")
    # self-check: the reader must round-trip its own fixture
    back = decode_example(next(iter_records(shard)))
    assert len(back["steps/is_first"]["int64"]) == k
    assert back["steps/is_first"]["int64"] == [1, 0, 0]
    assert back["steps/is_last"]["int64"] == [0, 0, 1]
    assert len(back["steps/action"]["float"]) == 7 * k
    assert back["steps/observation/image"]["bytes"][0][:4].hex() == "89504e47"
    print("fixture round-trip OK")


if __name__ == "__main__":
    main(sys.argv[1])
