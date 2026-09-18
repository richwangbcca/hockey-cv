# Labeling application guide

`labeling.app` is an offline annotation tool for the 2–4 Hz positioning dataset.
It serves a local browser interface; it does not send frames to an external service.
It requires Python, NumPy, and OpenCV. The interface has no npm dependencies.

## Open the prepared dataset

```sh
arch -x86_64 python3 -m labeling.app review-shots --dataset datasets/ice_v1
```

Open **http://127.0.0.1:8765**. The four phase buttons are independent passes:

1. **Choose usable frames:** click **Usable** when the view has enough rink context
   to place at least some visible on-ice people; otherwise click **Unusable**. Either
   choice saves, completes this phase, and advances. Players outside the frame do
   not matter. Formerly partial frames count as usable.
2. **Mark landmarks:** on selected ice views, choose a named dot or intersection
   and click it. The center line has three useful named points: its center dot and
   its intersections with each straight side board. For a visible stretch of line
   without a fixed intersection, use the optional `center_line` trace instead of
   inventing a point. Finish the visible landmarks with **Landmarks checked**. Only this
   phase shows suggestions. Optional line traces and ignore areas are collapsed.
3. **Check rink fit:** select extra landmarks as check points, define orientation,
   calculate the fit, and inspect the overlay. Choose accepted, insufficient,
   ambiguous, or rejected, then **Save fit review**. Marking landmarks and accepting
   a calibration are separate tasks with separate completion states.
4. **Mark players:** draw each on-ice person's box and add a visible skate-contact
   point. Finish with **Players checked**. Derived positions are an optional preview.

Each later-phase completion button saves and moves to the next frame in the current
filter. **Skip for now** saves a draft without completing it. **Unfinished in this phase**
uses only the current phase's status. You can change phases at any time; finishing
the entire pool is not required before exploring later phases on selected frames.
Use the frame picker/search to choose those frames; later phases do not automatically
exclude non-ice frames.

Sequence IDs, replay groups, and technical frame details are under **Optional notes
& frame details**. Most frames do not need them during the first pass. Landmark
names are written out in the picker; E/W and N/S still follow the diagram's axes.
Click **Add another landmark/person** to stop editing the selected item and start
a new one. The normal landmark tools are point placement, moving a point, and pan;
optional tracing/ignore actions appear when explicitly selected.

Existing annotations are preserved. Old completed geometry reviews count as
completed landmark reviews too. Moving an observed landmark reopens both reviews;
changing only its check-point reservation reopens the fit review. Observed features
can be exported once the landmark phase is complete, even before calibration.

`review-geometry` and `review-players` start in phases 2 and 4 respectively.
Stop the server with Ctrl+C. The `arch -x86_64` prefix is specific to this Mac's
Intel NumPy/OpenCV installation; use `python3` normally in a matching environment.

The prepared manifest contains:

- 1,175 images from `pool/` under source ID `broadcast_01`.
- The other 29 broadcast images from `benchmarking/frames/`, with the same source
  ID. These filenames are absent from the pool; originals stay in their folders.
- Seven original PNGs from `frames/`, under `regression_frames`, excluding overlays.
- 239 timestamped frames from `MINvSJS.mp4`, sampled every 20 seconds under source
  ID `min_vs_sjs`.
- 271 timestamped frames from `TORvTBL.mp4`, sampled every 20 seconds under source
  ID `tor_vs_tbl`.
- 197 legacy landmark proposals across the 36 previously labeled images. They
  remain unreviewed until visually accepted. Source image hashes must match on import.

The dataset now contains 1,721 images. All usability decisions have been reviewed;
geometry and player labels remain separate later phases. Indexing alone does not
claim that newly added images have been labeled. Dataset files
are ignored by Git; back up `datasets/ice_v1/` separately along with source images.
Manifest image paths are absolute; reindex a relocated source with the same source
ID and relative filenames to update paths without replacing annotations.

## Annotation workflow

1. **Choose usable frames.** Click Usable when at least some visible on-ice people
   could be mapped from the view; click Unusable for close-ups, benches, graphics,
   or ice views without enough rink context. The click saves and advances.
2. **Review rink geometry.** Accept or reject old/proposed points individually.
   Use the rink picker or landmark selector, select Landmark, and click the image.
   Use Add another landmark before choosing defaults for another annotation;
   changing fields while an annotation is selected edits that annotation.
   Polylines/arcs require a non-landmark feature type, clicks for vertices, and
   Enter to finish. Use separate segments across occlusions. Record unseen
   landmarks with visibility `occluded` or `out_of_frame`, without coordinates.
3. **Check calibration.** Define the orientation and document it in Notes.
   Use at least four nondegenerate observed point landmarks to fit; mark other
   independent points Held out. Save, then Fit. Review the overlay and held-out
   pixel/feet errors before selecting geometry status `accepted`. No accuracy
   threshold has been established: acceptance is an explicit human assessment.
   Finally click Save fit review. Insufficient or ambiguous views can
   still have reviewed image-space feature labels.
4. **Label players.** Drag a box around a visible on-ice person and set their role.
   Select the box in Annotations, set contact visibility, then select Player ice
   point and click the skate-contact midpoint. Hidden contacts have no point;
   uncertain or estimated contacts must be tagged accordingly. Set occlusion and
   truncation flags. Draw ignore polygons for ambiguous bench/graphic regions.
   Label officials separately. Mark player review `reviewed` only after checking
   all visible on-ice people; a reviewed empty list is a negative example. The
   Players checked button handles this status for you.
5. **Validate and export.** Review states are separate, so shot-only annotations
   can be exported before detailed geometry/player work is complete.

Wheel zooms; drag in Inspect pans. Select a feature and use Move selected point
to reposition the nearest vertex. Delete removes selected annotations. Undo
restores recent edits in the current session. Escape cancels unfinished drawing.
Autosave runs after edits; navigation saves first and stops on errors. Save status
and errors are visible. Reload resumes the last image. Browser-close warnings
protect unsaved changes, but cannot recover edits after a browser crash.

Observed geometry changes invalidate calibration. Player edits reset player review.
The green dashed template is inferred; its polygon encloses the fit anchors and
marks the supported region. The overlay is not additional observed truth. Rink
coordinates are derived only for accepted, reviewed calibration, visible contact
points, and points inside that region. Extrapolated, estimated, and hidden positions
remain null in derived output.

## CLI reference

Run index/import/export only when nobody is editing the dataset. One server process
may serve multiple tabs; revision checks prevent stale tabs from overwriting each
other. Simultaneous independent server/CLI writers are not supported.

```sh
python3 -m labeling.app index pool --dataset datasets/ice_v1 --source-id broadcast_01
python3 -m labeling.app index benchmarking/frames --dataset datasets/ice_v1 --source-id broadcast_01
python3 -m labeling.app index frames --dataset datasets/ice_v1 --source-id regression_frames
python3 -m labeling.app index broadcasts/min_vs_sjs/frames --dataset datasets/ice_v1 --source-id min_vs_sjs --timestamps broadcasts/min_vs_sjs/timing.json
python3 -m labeling.app index broadcasts/tor_vs_tbl/frames --dataset datasets/ice_v1 --source-id tor_vs_tbl --timestamps broadcasts/tor_vs_tbl/timing.json

python3 -m labeling.app import-legacy --dataset datasets/ice_v1 --labels benchmarking/label --images benchmarking/frames
python3 -m labeling.app import-legacy --dataset datasets/ice_v1 --labels labels --images frames
python3 -m labeling.app contact-sheet --dataset datasets/ice_v1 --out datasets/ice_v1/contact_sheets
python3 -m labeling.app validate --dataset datasets/ice_v1
python3 -m labeling.app export --dataset datasets/ice_v1 --out datasets/ice_v1/exports/draft_01
python3 -m labeling.app sample --dataset datasets/ice_v1 --hz 2 4 --out datasets/ice_v1/samples.json
```

`propose --dataset … --image-id <manifest ID> --method center` proposes neutral-zone
dots when the 601 detector can solve the view. `--method general` proposes unassigned
paint segments on frames tagged `ice_action`. Both are also available as buttons.
General proposals use the older detector and can be wrong; they are only review aids.
Accepted proposals retain `original_provenance` and their import source, if any.

For timing, index supports `--timestamps timing.json`, a mapping from relative
image filenames to confirmed source metadata, for example:

```json
{
  "f_0001.jpg": {
    "source_video": "broadcast-original.mp4",
    "source_frame_index": 300,
    "timestamp_seconds": 10.0
  }
}
```

The example is illustrative, not this pool's extraction timing. The sample command
requires timestamps and confirmed shot IDs. It selects available frames near a
common 2/4 Hz time grid, without interpolating or filling gaps. With the current
unknown timing it correctly produces empty selections and a skipped-image list.
Video extraction and temporal propagation remain deferred until timing is known.

## Storage and exports

`manifest.json` stores immutable image hashes, dimensions, IDs, and source timing.
`records/<image ID>.json` stores versioned observations, proposals, review states,
orientation, optional fit, and revision. Writes replace files atomically.

Export creates `shots.jsonl`, `features.jsonl`, `players.jsonl`, `calibrations.jsonl`,
`collector.json`, and `audit.json`. The first three include only their reviewed
annotation layer; inferred/pending feature proposals are excluded. `collector.json`
maps image IDs to the existing landmark-name/point format; the manifest provides
the original filename mapping. Exporting again to the same folder refreshes those
files, so use a new output directory to preserve an evaluation snapshot.

Splits group confirmed shots, explicit replay/action links, and exact image
duplicates. Unknown groups remain `unassigned`, rather than receiving random
frame-level splits. Group assignments are deterministic for a fixed manifest but
can change as groups/images are added; freeze an export before evaluation. Near
duplicates and unrecognized replays still require human grouping. These data only
measure within-broadcast performance, not generalization to other broadcasts.

## Verification and current limits

```sh
python3 -m unittest discover -s tests -v
node --check labeling/web/app.js
```

`python3 tests/browser_fixture.py` creates a fresh disposable fixture and prints
its dataset path. Serve that path on port 8766 for the test below.

`tests/browser_smoke.mjs` drives a real Chrome page through its local debugging
port, with no npm packages. It expects a **disposable** one-image dataset containing
601 and its eight imported legacy proposals, served at `127.0.0.1:8766`, and a
temporary Chrome profile launched with `--headless --remote-debugging-port=9223`.
Run `node tests/browser_smoke.mjs`; it edits that fixture, checks reload persistence,
and writes `/tmp/homography-labeling-browser.png`. Never point it at working labels.
Recreate the fixture before rerunning. URLs can be set through `LABEL_TEST_URL`
and `CHROME_DEBUG_URL`.

This implements the manual review and dataset pipeline across arbitrary views,
with limited automated assistance. Calibration currently uses point landmarks;
annotated arcs/lines are saved for training and visual inspection but are not yet
constraints in the solver. Near-duplicate detection, automatic shot classification,
distortion correction, and player proposals are not implemented. Split screens
should be marked unusable in this version. The rink template inherited from
`vision.rink` is explicitly provisional pending the planned dimension/convention audit.
