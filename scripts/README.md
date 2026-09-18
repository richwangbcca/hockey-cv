# Utility scripts

These are one-off data-preparation and completed pilot-workflow tools. They are
kept outside the project root so the active labeling and image-processing entry
points remain easy to identify.

- `prepare_broadcasts.py` extracts sparse, timestamped frames from broadcasts.
- `grab.py` extracts checkpoints for propagation drift experiments.
- `benchmark.py` collects legacy landmark clicks and scores homography error.
- `pilot/` contains the scene-classification and dataset-population pilot tools.
  Their reports, review files, and generated outputs remain in `../pilot/`.

Run these commands from the repository root, for example:

```sh
python3 scripts/prepare_broadcasts.py --help
python3 scripts/grab.py VIDEO.mp4
python3 scripts/pilot/score_review.py
```
