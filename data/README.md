# Data

Large source media and working labels live here. Most of this directory is local
and ignored by Git; back it up separately. The reviewed dataset in
`datasets/ice_v1/` and its referenced source images must be kept together.

| Folder | Purpose |
| --- | --- |
| `videos/` | Original broadcasts and short test clips. |
| `pool/` | Original frame pool from the first broadcast. |
| `broadcasts/` | Sparse extracted frames plus timing metadata for the other broadcasts. |
| `frames/` | Seven original regression frames and generated overlays. |
| `labels/` | Legacy hand-clicked landmarks for regression frames. |
| `benchmarking/` | Earlier benchmark frames and tracked legacy landmark JSON files. |
| `datasets/ice_v1/` | Current manifest, per-image records, selections, and exports. |
| `pilot/` | Scene-classification pilot report, reviews, and generated outputs. |

The manifest currently stores absolute source-image paths. If this repository
moves to another computer or directory, reindex each source using its existing
source ID and relative filenames as described in the
[labeling guide](../docs/LABELING_GUIDE.md).
