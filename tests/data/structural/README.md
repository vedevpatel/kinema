# Structural regression pack

Small extracts of real LeRobot datasets with structural metadata defects, at
exact Hub revisions, tested offline on every push by
`tests/test_structural_pack.py`. Rebuild with
`python tests/data/structural/build_pack.py [case ...]` (needs the Hub).

**What an extract holds.** `meta/info.json` and `meta/tasks.parquet` verbatim;
`meta/episodes` trimmed to the kept episodes and to the pointer and
video-segment columns (per-episode statistics dropped); the kept data shards
reduced to index columns (`episode_index`, `frame_index`, and where small
enough `timestamp`, `index`, `task_index`) - no state, action or image values;
and `listing.txt`, the repo's file listing at the revision for `meta/`, every
data shard (packed or not - as on the board, the listing is the whole repo's)
and every video file the kept episodes name. `CASE.json` records
source, revision, license, what was kept, the ground truth and the evidence.

**How the expected findings were verified.** `build_pack.py` / `cases.py`
compute the truth from the Hub files at the pinned revision with plain
pyarrow - no kinema check code: each kept episode's declared shard against
that shard's actual `episode_index` roster; per camera, episodes whose segment
`chunk_index` is null; per camera, segment-named video files absent from the
listing; and episodes catalogued more than once. The test requires each
detector to report exactly that set.

**Licenses.** Only datasets whose card declares a license permitting
redistribution are packed (MIT and Apache-2.0 here); each extract is a
derivative excerpt redistributed under its dataset's license, with
attribution below. Datasets with no declared license are excluded even when
they carry useful defects: no real missing-video-file case is packed for
that reason (every such case on the board - the ShareLab `thinking_*` repos
and `OpenMOSS-Team/robodojo_lerobot_v3.0` - declares no license), so
video-file-pointers is exercised here only by the clean controls, and by the
synthetic tests and the nightly network suite.

| case | source (attribution) | revision | license | defect and why the expected finding is correct |
| --- | --- | --- | --- | --- |
| `vlabench_pointer_rollover` | [VLABench/vlabench_primitive_ft_lerobot_video](https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video) | `9846a2f6bead` | MIT | The v2.1->v3.0 converter rollover bug ([lerobot#2401](https://github.com/huggingface/lerobot/issues/2401)): boundary episodes point one shard early - episode 2 is catalogued in `file-000` while its rows are in `file-001`, 4 -> `file-002`, and so on. 26 of the 27 broken pointers in the first 40 episodes have their rows in a packed shard (reported as moved); the last, episode 39, is catalogued in `file-026` with its rows in `file-027`, which the extract does not pack (reported as no rows in the packed shards). Verified by reading the shards' rosters at the revision. |
| `bitrobot_pointer_rows_unpacked` | [BitRobot/berkeley-frodobots-lerobot-7k](https://huggingface.co/datasets/BitRobot/berkeley-frodobots-lerobot-7k) | `096912cad351` | Apache-2.0 | 83 episodes (2031...) are catalogued in `file-006`, which does not hold their rows. The extract packs `file-006` only, so the expected finding is "no rows in the packed shards". Evidence: reading all 236 shards' rosters at the revision found every one of the 83 in `file-002` - the same rollover pattern, not missing data. |
| `mendelbeans_pointer_no_rows` | [MendelBeans/lerobot_dataset_put_pen_in_the_holder](https://huggingface.co/datasets/MendelBeans/lerobot_dataset_put_pen_in_the_holder) | `0c0e6528c6b3` | Apache-2.0 | Episodes 80 and 81 point at `file-002` and their rows exist in no shard (all 3 read): the catalogue was committed ahead of the data. The catalogue also lists 30 episodes twice (a dropped-frames FAIL); 50 and 51 are listed twice with different shards, so they have no single declared shard and are left out of the pointer comparison. |
| `yaak_segments_missing` | [yaak-ai/L2D-v3](https://huggingface.co/datasets/yaak-ai/L2D-v3) | `c9a9d0713e25` | Apache-2.0 | Episodes 9761-9845 (85 of the 160 kept, 9700-9859) record no video segment for any of the 7 cameras: their `videos/<camera>/chunk_index` is null. 53% of the kept episodes, so camera-coverage WARNs (>= 10%). The board's 250-episode sample saw only 2 of them. |
| `pusht_clean` | [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) | `7628202a2180` | MIT | Clean control: every pointer matches its shard, every segment is present, every named video file is in the listing. |
| `svla_so101_clean` | [lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) | `f641879e2217` | Apache-2.0 | Clean control, as above, with two cameras. |

What this pack shows is detector correctness on real metadata structure. It
says nothing about training outcomes.
