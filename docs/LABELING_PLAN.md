# Dataset labeling plan

## Objective and scope

Build an offline, human-reviewed labeling script for broadcast frames from any
camera angle. Its output should support a future system that estimates anonymous
player positions on the rink at **2–4 Hz**. This plan covers dataset preparation,
shot categorization, rink-feature annotation, anonymous player-location annotation,
quality checks, and export. It does not cover training or deploying that system,
identifying players, tracking identities, or analyzing tactics.

“Any angle” means the tool can represent and review every view, including views
with insufficient information to recover rink coordinates. It must not promise
automatic calibration of a close-up, an obscured rink, or an ambiguous crop.
Automation proposes labels; reviewed observations are the source of truth.
The offline labeling tool does not need to run at the eventual inference rate.

## Starting dataset

- Start with `pool/`: 1,175 JPEGs from one broadcast, named between `f_0001.jpg`
  and `f_1204.jpg`, with gaps. The inspected first image is 1920×1080. A 24-frame
  contact sheet shows center-ice and end-zone views, lower angles, close-ups,
  benches, and off-ice footage. Inventory all dimensions and classes before
  assuming the sample represents the full pool.
- Preserve originals and import existing annotations from `benchmarking/label/`,
  `labels/`, and any other discovered label folders after inspecting their schemas.
  Match by source image and dimensions; do not silently merge filename collisions
  or assume old labels have been reviewed under the new conventions.
- Establish extraction provenance: source video, original frame index, and source
  timestamp where recoverable. A pool filename is an identifier, not proof of
  source timing. Missing timing remains explicitly unknown.
- Use the seven original `frames/` images, especially 601, as regression examples.
  Keep generated overlays out of the input dataset.
- This broadcast is a development dataset. Coverage across other arenas,
  broadcasts, camera setups, and rink artwork requires additional data later.

## Annotation contract

Store one versioned record per image, with separate review states for shot tags,
geometry, and players. Distinguish `unreviewed`, `in_progress`, and `reviewed`;
an empty reviewed annotation means something different from work not yet done.

### 1. Usability gate

The implemented first pass is deliberately binary. `usable` means the view has
enough rink context to recover positions for at least some visible on-ice people;
`unusable` covers close-ups, benches, ads, transitions, and ice views without
enough context. Players outside the frame do not affect the label. Earlier
`partial` labels map to `usable`.

The richer fields below remain available as legacy metadata but are not part of
the required labeling workflow.

### Legacy shot metadata

| Field | Proposed values | Purpose |
| --- | --- | --- |
| `content` | `ice_action`, `bench_or_penalty_box`, `person_closeup`, `crowd_or_arena`, `ad_or_fullscreen_graphic`, `studio_or_other`, `mixed_or_transition`, `unknown` | Includes negative examples for a future shot filter. |
| `camera_view` | `elevated_wide`, `true_overhead`, `end_or_corner`, `low_rinkside`, `tight`, `not_applicable`, `unknown` | Do not call the normal elevated sideline camera truly overhead. |
| `temporal_context` | `live`, `replay`, `unknown` | A replay can look exactly like a useful live shot. Use neighboring footage when available; otherwise retain uncertainty. |
| `position_usability` | `usable`, `partial`, `unusable`, `unknown` | Whether rink coordinates can be recovered for all, some, or none of the on-ice people visible in the frame. Players outside the frame do not affect this label. Record a reason for partial/unusable action shots. |
| `geometry_status` | `unreviewed`, `insufficient`, `ambiguous`, `accepted`, `rejected` | Separate a good camera view from a validated calibration. |

Use reasons such as `tight_crop`, `occlusion`, `blur`, `insufficient_landmarks`,
`orientation_ambiguous`, or `non_ice`. An elevated shot is not automatically usable;
a low angle is not automatically useless. Advertisements painted on the ice and
ordinary scorebugs do not turn game footage into an ad shot. Mark obstructing
graphics with optional ignore polygons. For split screens, permit separate view
regions, each with its own geometry, or mark the frame unusable in the first release.

### 2. Observed rink geometry

Support direct clicks and short polylines for:

- Center dot; neutral-zone and end-zone faceoff dots.
- Center, blue, and goal lines: annotate the center of the painted stripe, with
  a consistent template convention. Annotate visible segments separately across
  occlusions; do not trace through a player and call it visible paint.
- Faceoff-circle arcs and the center circle, using multiple visible points.
- Crease and trapezoid markings, and true ice/board intersections where visible.
- Goalpost bases only when their contact with the ice is identifiable. Net tops,
  glass, and board artwork are not points on the ice plane.

Each feature carries semantic type, optional rink-instance ID, image coordinates,
visibility (`visible`, `partially_occluded`, `occluded`, `out_of_frame`, `uncertain`),
and provenance (`manual`, `detector_proposal`, `propagated`, `model_inferred`).
Do not invent coordinates for unseen features. Allow a visible blue line or dot
to remain semantically ambiguous until rink orientation is established.

Use a versioned rink template based on `vision.rink`'s center-origin, feet-based
coordinates; audit its dimensions and stripe conventions before freezing it.
Keep rink axes fixed in rink space, not redefined as image-left/image-right after
every camera cut. If physical orientation cannot be established, record an
arbitrary but documented sequence-local orientation or unresolved symmetry;
do not imply that opposite-end views are aligned to a known global orientation.

### 3. Anonymous player locations

For each visible on-ice person, label a visible bounding box, role (`skater`,
`goalie`, `official`, `unknown`), and an ice-plane reference point when observable.
For upright skaters, use the midpoint of visible skate/ice contacts; with one
visible contact, record that fact and whether the midpoint is estimated. Never
use the box bottom as an automatic substitute. Establish example-based conventions
for goalies, kneeling/fallen players, and partial truncation during the pilot;
mark position uncertain when a consistent point cannot be placed.

Keep contact visibility, occlusion, truncation, and point provenance explicit.
Players with hidden feet can still have a detection box, but not a fabricated
ground-truth location. Exclude bench occupants from on-ice player labels; use
ignore regions for ambiguous bench areas. Label officials separately so they do
not become false skater targets. Annotation IDs are local record identifiers,
not player identities. Names, jersey numbers, team assignments, puck labels,
and cross-frame player tracks are outside this first version.

Image observations remain primary. Rink coordinates are derived only with an
accepted calibration and within its supported region; retain the calibration
version and uncertainty. Derived coordinates are not independent ground truth.

## Proposed script and review workflow

Provide a single entry point, `labeling.app`, with the following proposed commands.
The interface can extend the existing Python/OpenCV/Matplotlib collector, with
zoom, pan, undo, autosave, and resume; avoid building a large application initially.

1. **`index pool/ --dataset datasets/ice_v1`**: inventory image hashes, dimensions,
   provenance, duplicates, and existing labels. Generate contact sheets and a
   manifest. Preserve source paths and originals.
2. **`review-shots`**: rapidly tag every pool image, preferably in chronological
   order. Bulk-apply tags to confirmed runs with per-frame overrides. Propose shot
   boundaries only when source ordering is established; sparse frames cannot
   identify every intervening cut. Review transitions and replay status explicitly.
3. **`propose`**: generate optional feature candidates on ice views. Reuse
   `vision.detect_center` only for its supported center-ice configuration. Add general
   line, ellipse/arc, and dot candidates, then propose their correspondence to the
   rink template. Color is supporting evidence, not semantic identity: logos,
   uniforms, ads, and shadows are hard negatives. Save proposals separately.
4. **`review-geometry`**: show the image beside a rink diagram; select a feature
   on the diagram, then click or trace its observed image location. Permit any
   visible subset and unresolved identities. Fit a candidate homography from
   sufficient independent constraints and show the projected rink overlay.
   Require manual acceptance; allow `insufficient` without blocking other labels.
5. **`review-players`**: annotate anonymous instances and ice points on selected
   ice frames. Show any derived rink positions beside the image for review.
   Do not require valid rink geometry to save useful image-space annotations.
6. **`validate` / `export`**: validate schema, provenance, geometry, and review
   completeness; export separate shot, observed-feature, and player datasets.
   Export derived calibration data separately. Provide a compatibility export
   for reviewed point landmarks in the current `vision.rink` JSON format.

For optional propagation, restrict suggestions to confirmed continuous camera
shots with known source timing. Reset at cuts, replay transitions, or uncertain
continuity. Reuse `vision.propagate` only after restricting tracked features to static
ice regions and checking drift; do not carry moving players, crowd, or board
features as ice anchors. Propagated annotations need review and retain provenance.
Do not assume pool adjacency provides suitable optical-flow intervals.

## Geometry acceptance and dataset quality

- Four suitably arranged point correspondences can determine a homography, but
  fitting four points exactly is not independent validation. Reject degenerate
  layouts; consider feature spread and uncertainty, not just anchor count.
- Validate against additional observed dots, lines, or arcs not used to fit it.
  Where available, use held-out landmarks and report both image-pixel and rink-
  space errors. Set acceptance tolerances after the pilot, rather than presenting
  an arbitrary threshold as established player-position accuracy.
- Draw the full template for inspection, but mark extrapolated areas. A center-
  ice fit can look good locally and be inaccurate at the boards. Store the region
  over which calibration has supporting observations.
- Preserve alternative hypotheses for symmetric views until resolved. A visually
  plausible mirrored solution must not silently become a global rink mapping.
- Flag views with systematic overlay mismatch, including possible lens distortion;
  keep observed labels even if a single planar homography is inadequate. Any later
  distortion correction must preserve its transform back to original coordinates.
- Never promote inferred circle centers, projected dots, or model intersections
  into independent observed anchors. Keep pseudo-labels out of evaluation truth.
- Audit a second annotation pass on a subset to measure click disagreement,
  shot-tag consistency, missed people, and contact-point ambiguity. Report results
  by camera view and visibility, not only an aggregate score.

## Sampling and the 2–4 Hz goal

First categorize all 1,175 pool frames. Then choose a diverse geometry/player
subset across center ice, both ends, corners, zoom levels, partial views, crowded
play, motion blur, occlusion, and hard negative content. Deduplicate near-identical
images before spending time on detailed labels. Retain both a representative
sample and targeted hard examples so curated difficulty does not disguise natural
shot frequencies.

Once source timing is known, include short contiguous sequences sampled every
**500 ms (2 Hz)** and **250 ms (4 Hz)**. Share frames between those two sets when
possible. Include pans, zooms, and camera cuts, retaining timestamps and shot IDs.
If the pool lacks the needed cadence, extract these sequences from the source
video after confirming the mapping; do not interpolate images or invent timing.
These labels enable later comparisons of positional coverage and temporal
consistency at both rates. Selecting the runtime model or proving its latency is
outside this labeling project.

Split data by continuous shot or larger temporal group, not randomly by frame.
Group duplicates and identifiable replays of the same action together to reduce
leakage. The single-broadcast holdout measures within-broadcast performance only;
reserve other broadcasts for a later generalization test. Freeze evaluation labels
before tuning automation against them.

## Implementation stages and completion criteria

1. **Manifest and shot review:** index the pool, inspect legacy labels, document
   timing availability, and implement shot tags. Complete the first categorization
   pass and publish category counts and unknowns.
2. **Manual geometry pilot:** implement the template-backed editor and explicit
   visibility/provenance. Review roughly 50 diverse images, including unusable
   views. Freeze annotation conventions after resolving pilot disagreements.
3. **Player-point pilot:** add instance boxes and contact points on usable and
   partial ice views. Create a small visual annotation guide from hard cases;
   measure annotation time and disagreement before scaling.
4. **Assisted labeling:** add candidate geometry and, if worthwhile, player-box
   proposals. Measure proposal correction time against manual labeling. Add
   propagation only if source continuity is established and it reduces review work.
5. **Dataset release:** finish the selected detailed subset, run validation,
   freeze grouped splits, and export labels, overlays, manifests, and an audit report.

Completion means every pool image has reviewed shot tags; the selected detailed
subset has explicit geometry/player review states; uncertain and impossible views
are represented without fabricated coordinates; exports can be regenerated; and
the report states coverage, annotation agreement, missing timing, and the limits
of this single-broadcast dataset. It does not mean every angle yields a calibration.
