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

**v2.1 extracts** (one parquet and one MP4 per episode; the `fastumi_*`
cases, each one dataset folder of a multi-dataset repo, recorded as
`subdir`): `meta/info.json` and `meta/tasks.jsonl` verbatim,
`meta/episodes.jsonl` reduced to the kept episodes' lines byte-for-byte,
each kept episode's data parquet reduced to `episode_index` and
`frame_index`, `episodes_stats.jsonl` and all videos left out. The positive
case's `listing.txt` holds **every** `meta/` and `videos/` file of the
folder at the revision, kept episodes or not, so trimming the catalogue
cannot make a video look absent; the control's holds the videos its kept
episodes name, as the v3 extracts do. Data entries are listed for the kept
episodes only (no v2.1 check reads the others); whole-folder counts and the
full listing's sha256 are in `CASE.json` evidence. v2.1 records no
declared shard and no video segment, so episode-file-pointers and
camera-coverage have nothing to compare: `CASE.json` marks them
`not_applicable` and the test requires them to skip.

**How the expected findings were verified.** `build_pack.py` / `cases.py`
compute the truth from the Hub files at the pinned revision with plain
pyarrow - no kinema check code: each kept episode's declared shard against
that shard's actual `episode_index` roster; per camera, episodes whose segment
`chunk_index` is null; per camera, segment-named video files absent from the
listing (v2.1: `video_path` rendered per episode and camera); and episodes
catalogued more than once. The test requires each detector to report
exactly that set - for video-file-pointers every missing file, every
affected episode and the per-camera severity of the documented policy.

**Licenses.** Only datasets with a license permitting redistribution are
packed (MIT and Apache-2.0 here); each extract is a derivative excerpt
redistributed under its dataset's license, with attribution below. The
license normally comes from the dataset card. FastUMI-100K's card carries
no license field, so it is packed on the publisher's own grant instead,
which `build_pack.py` re-verifies on every rebuild (`LICENSE_GRANTS` in
`cases.py`): the README at the pinned revision states "This dataset is
released under the **Apache-2.0 License**."; the paper
([arXiv:2510.08022](https://arxiv.org/abs/2510.08022)) names
[github.com/MrKeee/FastUMI-100K](https://github.com/MrKeee/FastUMI-100K) as
its release link, whose README (commit `4d2bb3f23b52`) names this Hub repo
as the dataset with the same grant and whose `LICENSE` is the Apache-2.0
text. Neither repo has a NOTICE file. Datasets with no
license anywhere are excluded even when they carry useful defects: the
board's other missing-video-file cases (the ShareLab `thinking_*` repos and
`OpenMOSS-Team/robodojo_lerobot_v3.0`) have no card license, README or
license file.

**License notices.** Every case directory carries a `LICENSE` file, and its
`CASE.json` records where that text came from (`license_file`, pinned by
sha256), the attribution (`attribution`) and what the source itself
provides (`license_notice`). The files in each directory are modified
(trimmed) excerpts of the originals - rows kept, columns dropped, as
described above and in each `CASE.json` - as Apache-2.0 section 4(b) asks
to be stated. None of the six v3 sources has a LICENSE, NOTICE or copyright
line of its own; each declares its license only in the card YAML at the
pinned revision, which `build_pack.py --notices` re-checks.

- Apache-2.0 (`bitrobot_*`, `mendelbeans_*`, `yaak_*`, `svla_so101_clean`):
  the canonical text from apache.org; there is no NOTICE file to carry
  over (section 4(d)). The `fastumi_*` cases carry the publisher's own copy
  (the same text, without the leading blank line).
- MIT (`pusht_clean`, `vlabench_pointer_rollover`): MIT asks for the
  copyright notice to travel with copies, and the Hub repos give none. No
  holder is invented: each carries the MIT `LICENSE` of the upstream project
  its card cites, copyright line retained verbatim - Diffusion Policy
  (`real-stanford/diffusion_policy@5ba07ac6661d`: "Copyright (c) 2023
  Columbia Artificial Intelligence and Robotics Lab") and VLABench
  (`OpenMOSS/VLABench@cf588fe60c0c`: "Copyright (c) 2024 simpler-env", as
  published there).

`python tests/data/structural/build_pack.py --notices [case ...]` rewrites
the v3 cases' `LICENSE` and these `CASE.json` fields without re-extracting
any data; a full rebuild writes them too.

| case | source (attribution) | revision | license | defect and why the expected finding is correct |
| --- | --- | --- | --- | --- |
| `vlabench_pointer_rollover` | [VLABench/vlabench_primitive_ft_lerobot_video](https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video) - VLABench, Zhang et al., [arXiv:2412.18194](https://arxiv.org/abs/2412.18194) | `9846a2f6bead` | MIT | The v2.1->v3.0 converter rollover bug ([lerobot#2401](https://github.com/huggingface/lerobot/issues/2401)): boundary episodes point one shard early - episode 2 is catalogued in `file-000` while its rows are in `file-001`, 4 -> `file-002`, and so on. 26 of the 27 broken pointers in the first 40 episodes have their rows in a packed shard (reported as moved); the last, episode 39, is catalogued in `file-026` with its rows in `file-027`, which the extract does not pack (reported as no rows in the packed shards). Verified by reading the shards' rosters at the revision. |
| `bitrobot_pointer_rows_unpacked` | [BitRobot/berkeley-frodobots-lerobot-7k](https://huggingface.co/datasets/BitRobot/berkeley-frodobots-lerobot-7k) | `096912cad351` | Apache-2.0 | 83 episodes (2031...) are catalogued in `file-006`, which does not hold their rows. The extract packs `file-006` only, so the expected finding is "no rows in the packed shards". Evidence: reading all 236 shards' rosters at the revision found every one of the 83 in `file-002` - the same rollover pattern, not missing data. |
| `mendelbeans_pointer_no_rows` | [MendelBeans/lerobot_dataset_put_pen_in_the_holder](https://huggingface.co/datasets/MendelBeans/lerobot_dataset_put_pen_in_the_holder) | `0c0e6528c6b3` | Apache-2.0 | Episodes 80 and 81 point at `file-002` and their rows exist in no shard (all 3 read): the catalogue was committed ahead of the data. The catalogue also lists 30 episodes twice (a dropped-frames FAIL); 50 and 51 are listed twice with different shards, so they have no single declared shard and are left out of the pointer comparison. |
| `yaak_segments_missing` | [yaak-ai/L2D-v3](https://huggingface.co/datasets/yaak-ai/L2D-v3) - L2D by [Yaak](https://www.yaak.ai/) | `c9a9d0713e25` | Apache-2.0 | Episodes 9761-9845 (85 of the 160 kept, 9700-9859) record no video segment for any of the 7 cameras: their `videos/<camera>/chunk_index` is null. 53% of the kept episodes, so camera-coverage WARNs (>= 10%). The board's 250-episode sample saw only 2 of them. |
| `fastumi_video_files_missing` | [IPEC-COMMUNITY/FastUMI_100k_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot), folder `single_arm/heat_food_in_microwave` - FastUMI-100K, Liu et al., [arXiv:2510.08022](https://arxiv.org/abs/2510.08022) | `4cf7cca8fdc1` | Apache-2.0 (publisher grant, see above) | Missing video files, v2.1. The folder catalogues 3156 episodes and holds all 3156 data parquets, but only 560 of the 3156 MP4s its `video_path` names: `videos/chunk-001/` and `videos/chunk-002/` do not exist and chunk-000 lacks 596 (gaps from episode 189 on, an interrupted upload). Verified on the paginated tree listing (the `dataset_info` siblings stop at 100000 for this repo) and again per path with `get_paths_info`, with present files as controls; one branch, no tags, no case-insensitive or sibling-folder copies; public, not gated. 39 of the 71 kept episodes (186-242, 287-292, 998-1001, 2998-3001) miss their MP4: 55%, so FAIL (>= 50%), as on the whole folder (82%). |
| `pusht_clean` | [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) - from Diffusion Policy, Chi et al., IJRR 2024 ([project](https://diffusion-policy.cs.columbia.edu/)) | `7628202a2180` | MIT | Clean control: every pointer matches its shard, every segment is present, every named video file is in the listing. |
| `svla_so101_clean` | [lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) | `f641879e2217` | Apache-2.0 | Clean control, as above, with two cameras. |
| `fastumi_v21_clean` | [IPEC-COMMUNITY/FastUMI_100k_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot), folder `single_arm/iron_clothes` - FastUMI-100K, Liu et al., [arXiv:2510.08022](https://arxiv.org/abs/2510.08022) | `4cf7cca8fdc1` | Apache-2.0 (publisher grant, see above) | v2.1 clean control from the same release: all 2541 MP4s its metadata names are in the folder listing (1000 + 1000 + 541). 34 kept episodes across all three chunks and both chunk edges. |

What this pack shows is detector correctness on real metadata structure. It
says nothing about training outcomes.
