# 100-frame automation pilot

## Status

The model pass is complete. Independent human review is still required before
these numbers can be called accuracy. Model confidence measures triage value; it
is not a substitute for comparison with human labels.

## Sample

The pilot contains 100 images from the 1,175-frame `data/pool/`. Selection divides the
broadcast into ten temporal blocks and chooses ten visual cluster medoids within
each block. This keeps coverage across the game and deliberately favors visual
variety. The resulting class frequencies do **not** estimate natural broadcast
frequency: rare close-ups and non-ice views are intentionally overrepresented.

The scene pass used contact sheets at 480×270 pixels per frame. It assigned the
existing content, camera-view, temporal-context, and position-usability classes.
Usability concerns only on-ice people visible in the frame.

## Model output

| Field | Predicted distribution | High-confidence labels | Labels flagged for review |
| --- | --- | ---: | ---: |
| Content | 67 ice action; 19 person close-up; 6 bench; 5 crowd; 2 other; 1 mixed | 89 | 11 |
| Camera view | 29 elevated wide; 24 end/corner; 22 tight; 6 overhead; 5 low; 14 N/A | 91 | 9 |
| Position usability | 62 usable; 38 unusable | 92 | 8 |
| Live/replay | 85 live; 6 replay; 9 unknown | 0 | 100 |

The user independently reviewed the binary usable/unusable result and reported
all 100 decisions correct. On this deliberately diverse single-broadcast sample,
that is **100/100 agreement**. It does not yet measure another broadcast or arena.

The clearest automation candidate is binary usability. Content and camera view
were recorded during the experiment but are no longer required by the workflow.
well suited to proposals, with human review focused on low-vs-tight and unusual
angles. Usability is straightforward for wide and non-ice scenes; the eight
flagged views require judgment about whether their visible people have enough rink
context. Live/replay should be derived from temporal shot context and broadcast
transitions, not accepted from isolated-frame predictions.

Across content, camera view, and usability, 84 frames have high model confidence
for all three fields; 16 have at least one field flagged. This is a proposed review
queue, not evidence that the other 84 are correct. A random audit of confident
predictions is still necessary.

## Geometry result

The conservative center-ice detector was run on all 67 predicted ice-action frames.
It returned one result (`f_0836.jpg`) and abstained on 66: **1.5% coverage**. Visual
inspection of the accepted overlay found all four neutral-zone dots and both blue
lines correctly aligned, with a reported 3.0-pixel mean blue-line discrepancy.

This is evidence of high selectivity, not a measured precision rate: one accepted
case is too small to estimate precision. It also shows that the current detector
cannot supply general-angle landmark labels. General geometry automation needs a
different method, likely semantic line/dot detection plus rink-template fitting,
and should continue to expose confidence and abstention.

## Player-location result

This environment has no installed hockey/player detector. The vision-model pass
can distinguish people and judge whether skate contacts appear visible, but contact
sheets are not an appropriate interface for precise boxes or pixel coordinates.
No player boxes or skate points were generated from this pilot, and no accuracy
claim is made for them. The next player pilot needs a detector that returns native
image coordinates, followed by manual correction on a small reviewed subset.

## What this pilot supports

- Generate content and camera-view proposals for the full pool, then review all
  uncertain predictions and a random sample of confident ones.
- Treat usability as a proposed label; fully review tight and low views.
- Group frames into continuous shots before automating live/replay.
- Reuse a validated homography within a continuous camera shot, rather than label
  landmarks independently on every frame.
- Keep current center-ice geometry proposals as a narrow, conservative fast path.
- Do not auto-accept precise landmarks or skate-contact positions based on this
  pilot; neither has independent ground truth here.

## Independent review

`output/review.csv` contains the 100 predictions and empty `reviewer_*` columns.
The nine `output/labeled_*.jpg` sheets show the same predictions under each image;
orange `CHECK` text identifies fields the model considered uncertain. Reviewers
should correct every row for a real accuracy estimate, or at minimum all flagged
fields plus a random sample of high-confidence fields. Once reviewer columns are
filled, run `python3 scripts/pilot/score_review.py` for per-field accuracy and confusion
counts without touching the working dataset.

`output/pilot_labels.json` is the machine-readable model output. It is separate
from `data/datasets/ice_v1/records`; this pilot has not marked any production annotation
as reviewed.
