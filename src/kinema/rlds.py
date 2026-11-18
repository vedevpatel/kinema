"""RLDS reader: anonymous GCS access + pure-Python TFRecord / tf.Example decoding.

RLDS datasets (Open X-Embodiment) live as TFDS-prepared directories on GCS:
<dataset>/<version>/{dataset_info.json, features.json, *-SPLIT.tfrecord-NNNNN-of-MMMMM}.
No tensorflow / tensorflow_datasets dependency: TFRecord framing is
length-prefixed with CRC32C, and tf.Example's protobuf schema has been frozen
since 2015, so both decode in stdlib Python. All network access is anonymous:
no auth headers are ever sent.
"""

from __future__ import annotations

import json
import struct
import time
import urllib.parse
import urllib.request

GCS_API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_MEDIA = "https://storage.googleapis.com/{bucket}/{name}"
DEFAULT_TIMEOUT = (10, 30)  # connect, read seconds
MAX_RETRIES = 3


class RldsError(Exception):
    """Any RLDS access or decode failure; callers treat a lane failure as empty."""


# --- CRC32C (Castagnoli) -------------------------------------------------
# zlib.crc32 is the wrong polynomial; TFRecord uses CRC32C with the mask below.

_CRC32C_TABLE: list[int] | None = None


def _crc32c_table() -> list[int]:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        poly = 0x82F63B78  # reflected Castagnoli
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                crc = (crc >> 1) ^ poly if crc & 1 else crc >> 1
            table.append(crc)
        _CRC32C_TABLE = table
    return _CRC32C_TABLE


def crc32c(data: bytes) -> int:
    table = _crc32c_table()
    crc = 0xFFFFFFFF
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return ((crc >> 15) | (crc << 17)) + 0xA282EAD8 & 0xFFFFFFFF


# --- TFRecord framing -----------------------------------------------------

def iter_records(buf: bytes):
    """Yield payloads from a TFRecord byte stream.

    Stops cleanly at the first incomplete trailing record so range-fetched
    heads work; raises RldsError on a corrupt length or CRC so damaged shards
    surface instead of decoding as garbage.
    """
    off = 0
    while off + 12 <= len(buf):
        (length,) = struct.unpack_from("<Q", buf, off)
        if off + 12 + length + 4 > len(buf):
            return  # incomplete trailing record (range fetch boundary)
        if _masked_crc32c(buf[off:off + 8]) != struct.unpack_from("<I", buf, off + 8)[0]:
            raise RldsError(f"tfrecord length CRC mismatch at offset {off}")
        payload = buf[off + 12:off + 12 + length]
        (want,) = struct.unpack_from("<I", buf, off + 12 + length)
        if _masked_crc32c(payload) != want:
            raise RldsError(f"tfrecord payload CRC mismatch at offset {off}")
        yield payload
        off += 12 + length + 4


def count_complete_records(buf: bytes) -> int:
    return sum(1 for _ in iter_records(buf))


# --- tf.Example wire decoding --------------------------------------------
# tensorflow/core/example/example.proto (frozen):
#   Example { Features features = 1; }
#   Features { map<string, Feature> feature = 1; }
#   Feature { BytesList bytes_list = 1; FloatList float_list = 2; Int64List int64_list = 3; }
#   XxxList { repeated T value = 1 [packed for scalars]; }

def _read_varint(buf: bytes, off: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if off >= len(buf):
            raise RldsError("truncated varint")
        b = buf[off]
        off += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, off
        shift += 7
        if shift > 70:
            raise RldsError("varint too long")


def _read_fields(buf: bytes):
    """Yield (field_number, wire_type, value) for a protobuf message."""
    off = 0
    while off < len(buf):
        key, off = _read_varint(buf, off)
        field, wire = key >> 3, key & 7
        value: int | bytes
        if wire == 0:
            value, off = _read_varint(buf, off)
        elif wire == 2:
            length, off = _read_varint(buf, off)
            value = buf[off:off + length]
            if len(value) != length:
                raise RldsError("truncated length-delimited field")
            off += length
        elif wire == 5:
            value = buf[off:off + 4]
            off += 4
        elif wire == 1:
            value = buf[off:off + 8]
            off += 8
        else:
            raise RldsError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _decode_list(buf: bytes, kind: str) -> list:
    out: list = []
    for field, wire, value in _read_fields(buf):
        if field != 1:
            continue
        if kind == "bytes":
            out.append(value)
        elif kind == "int64":
            if wire == 2:  # packed
                off = 0
                while off < len(value):
                    v, off = _read_varint(value, off)
                    out.append(v - (1 << 64) if v >= 1 << 63 else v)
            else:
                v = value
                out.append(v - (1 << 64) if v >= 1 << 63 else v)
        elif kind == "float":
            if wire == 2:  # packed
                for i in range(0, len(value), 4):
                    out.append(struct.unpack_from("<f", value, i)[0])
            else:
                out.append(struct.unpack("<f", value)[0])
    return out


def decode_example(buf: bytes) -> dict[str, dict[str, list]]:
    """Decode one serialized tf.Example into {name: {kind: values}}.

    Each value is {"bytes": [...]} or {"int64": [...]} or {"float": [...]} -
    exactly one key, matching the Feature oneof.
    """
    features: dict[str, dict[str, list]] = {}
    for field, wire, example_val in _read_fields(buf):
        if field != 1 or wire != 2:
            continue
        for f2, w2, features_val in _read_fields(example_val):
            if f2 != 1 or w2 != 2:
                continue
            name: str | None = None
            for f3, w3, entry_val in _read_fields(features_val):
                if f3 == 1 and w3 == 2:
                    name = entry_val.decode("utf-8", "replace")
                elif f3 == 2 and w3 == 2 and name is not None:
                    for f4, w4, feature_val in _read_fields(entry_val):
                        if f4 == 1 and w4 == 2:
                            features[name] = {"bytes": _decode_list(feature_val, "bytes")}
                        elif f4 == 2 and w4 == 2:
                            features[name] = {"float": _decode_list(feature_val, "float")}
                        elif f4 == 3 and w4 == 2:
                            features[name] = {"int64": _decode_list(feature_val, "int64")}
    return features


# --- anonymous GCS client --------------------------------------------------

class GcsClient:
    """Anonymous Google Cloud Storage JSON-API client. Never sends credentials."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.calls = 0

    def _get(self, url: str, byte_range: tuple[int, int] | None = None) -> bytes:
        assert "Authorization" not in url  # anonymous-only reader
        req = urllib.request.Request(url)
        if byte_range is not None:
            req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT[1]) as r:
                    self.calls += 1
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code < 500:  # 404/403 are answers, not transient failures
                    raise RldsError(f"GCS GET {e.code}: {url.split('?')[0][-80:]}")
                last = e
            except Exception as e:  # noqa: BLE001 - network errors are retried
                last = e
            time.sleep(0.5 * (attempt + 1))
        raise RldsError(f"GCS GET failed after {MAX_RETRIES} tries: {type(last).__name__}")

    def list(self, prefix: str = "", delimiter: str = "") -> tuple[list[str], list[tuple[str, int]]]:
        """-> (child prefixes, [(object name, size)]) across all pages."""
        prefixes: list[str] = []
        items: list[tuple[str, int]] = []
        token: str | None = ""
        while token is not None:
            q = {"prefix": prefix, "maxResults": "1000"}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["pageToken"] = token
            url = GCS_API.format(bucket=self.bucket) + "?" + urllib.parse.urlencode(q)
            page = json.loads(self._get(url))
            prefixes.extend(page.get("prefixes", []))
            for it in page.get("items", []):
                items.append((it["name"], int(it["size"])))
            token = page.get("nextPageToken")
        return prefixes, items

    def get(self, name: str, byte_range: tuple[int, int] | None = None) -> bytes:
        url = GCS_MEDIA.format(bucket=self.bucket,
                               name=urllib.parse.quote(name, safe=""))
        return self._get(url, byte_range)

    def get_json(self, name: str):
        return json.loads(self.get(name))


# --- dataset probe ----------------------------------------------------------

class RldsDataset:
    """One probed RLDS dataset: metadata + shard inventory, no payloads."""

    def __init__(self, name: str, version: str, info: dict, features: dict,
                 shards: list[tuple[str, int]]):
        self.name = name
        self.version = version
        self.info = info
        self.features = features
        self.shards = shards  # [(object name, size)]

    @property
    def data_bytes(self) -> int:
        return sum(s for _, s in self.shards)

    @property
    def splits(self) -> dict[str, int]:
        """split name -> declared episode count.

        Real TFDS dataset_info.json has no numExamples: episode counts live in
        shardLengths (one entry per shard, strings). Fall back to numExamples
        for other TFDS flavors.
        """
        out: dict[str, int] = {}
        for s in self.info.get("splits", []):
            if "shardLengths" in s:
                out[s.get("name", "")] = sum(int(x) for x in s["shardLengths"])
            else:
                out[s.get("name", "")] = int(s.get("numExamples", 0) or 0)
        return out

    @property
    def shard_lengths(self) -> dict[str, list[int]]:
        """split name -> declared episodes per shard (empty when undeclared)."""
        return {s.get("name", ""): [int(x) for x in s.get("shardLengths", [])]
                for s in self.info.get("splits", [])}

    @property
    def episodes(self) -> int:
        return sum(self.splits.values())

    @classmethod
    def probe(cls, client: GcsClient, dataset_prefix: str) -> "RldsDataset":
        """Probe <prefix>/ (holding version dirs); picks the newest TFDS semver.

        GCS has no directories: version dirs surface either as listing
        prefixes or as `<version>_$folder$` placeholder objects, so both
        shapes are collected.
        """
        name = dataset_prefix.rstrip("/").split("/")[-1]
        prefixes, items = client.list(prefix=dataset_prefix.rstrip("/") + "/",
                                      delimiter="/")
        candidates = [p.rstrip("/").split("/")[-1] for p in prefixes]
        base = dataset_prefix.rstrip("/") + "/"
        for n, _ in items:
            leaf = n[len(base):]
            if leaf.endswith("_$folder$"):
                candidates.append(leaf[: -len("_$folder$")])
        versions = [v for v in candidates if _semver_key(v) is not None]
        if not versions:
            raise RldsError(f"{name}: no versioned directories")
        version = max(versions, key=_semver_key)  # type: ignore[arg-type]
        vprefix = f"{dataset_prefix.rstrip('/')}/{version}/"
        _, items = client.list(prefix=vprefix)
        names = {n.rsplit("/", 1)[-1] for n, _ in items}
        # a version dir with no metadata holds no data (44 of 73 prefixes in
        # the Open X bucket are _$folder$ markers with empty version dirs -
        # their data never landed here); skip fast, no doomed fetches
        missing = [f for f in ("dataset_info.json", "features.json") if f not in names]
        if missing:
            raise RldsError(f"{name}/{version}: no {missing[0]} - no data in this bucket")
        info = client.get_json(vprefix + "dataset_info.json")
        features = client.get_json(vprefix + "features.json")
        shards = [(n, s) for n, s in items if ".tfrecord-" in n or ".array_record-" in n]
        return cls(name, version, info, features, shards)


def _semver_key(v: str) -> tuple[int, int, int] | None:
    parts = v.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)  # type: ignore[return-value]
