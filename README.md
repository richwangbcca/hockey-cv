# A CV-based Bandage for NHL Coordinate Data

## Problem

The NHL does not expose raw player coordinate data. You can get so much data from NHL EDGE, but
you can't get the (x, y) positions of players on the ice.

## Proposed solution

A computer-vision and machine-learning pipeline that watches a game broadcast
live and estimates player coordinates in rink-space.

It detects on-ice landmarks — lines and faceoff dots — and uses them to compute
a homography that maps broadcast-camera space onto a real rink coordinate
system. Player detections are then projected through that homography to recover
their positions on the ice.

## Current status
A testing harness was made to estimate whether homography would even be possible.
Players often times occlude on-ice landmarks, and a frame needs four landmarks to
be able to translate broadcast-space to rink-space. The test seemed reasonable, and
I am now building the actual homography computation engine.

## Data labeling

The local labeling interface supports binary shot usability, rink landmarks and
polylines, anonymous player boxes/contact points, reviewed calibration, and dataset export.
See the [labeling guide](docs/LABELING_GUIDE.md) for setup and controls, and
the [labeling plan](docs/LABELING_PLAN.md) for the 2–4 Hz dataset scope.

```sh
arch -x86_64 python3 -m labeling.app review-shots --dataset datasets/ice_v1
```

Then open http://127.0.0.1:8765. The prepared dataset contains 1,721 reviewed
usability labels: 1,714 across three broadcasts plus seven original regression frames.
The two newer broadcasts were sampled every 20 seconds and retain source timestamps.

## Center-ice labeling experiment

`python3 -m vision.detect_center frames/601.png` produces `frames/601_v3.png`,
`601_v3.labels.json` (four observed neutral-zone dots, in the collector's format),
and `601_v3.report.json` (line segments, inferred center, and local homography).
On the current Mac installation, use `arch -x86_64 python3` because the installed
NumPy/OpenCV binaries are Intel builds.

This detector accepts foreshortened, pale red ellipses and selects a four-dot
configuration using independent blue-line alignment. It fits the center stripe
from nearby red pixels. The dashed center circle and diamond are inferred from
dot geometry, not additional independent homography correspondences. Original
images, hand labels, and the v2 detector are preserved; no hand labels are read
by the new detector.

On 601, all four neutral-zone dots match the existing hand labels within 8.5 px
(mean 6.6 px); v2 matches zero within 18 px and emits 19 dot candidates. The
inferred center differs from its hand label by 2.3 px. Hand clicks are approximate,
so these are agreement measurements, not subpixel ground-truth accuracy.

This is a shot-specific center-ice prototype, tuned on 601, not a general rink
detector. It requires two blue lines and four visible neutral-zone dots. It
declines all six other original frames in this folder, as well as blank and
grayscale inputs, rather than emitting an unsupported configuration. The
four-dot homography has a 15.7 px mean discrepancy against the detected blue
lines; do not extrapolate it to board intersections or assume calibrated
player-coordinate accuracy.
