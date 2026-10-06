# Vision experiments

`rink.py` defines the provisional rink template. `detect.py` and
`detect_center.py` contain landmark detectors; `propagate.py` is an experimental
shot-level propagation tool. These modules are separate from the manual labeling
server and can be run from the repository root, for example:

```sh
arch -x86_64 python3 -m vision.detect_center data/frames/601.png
```

See the root [README](../README.md) for the detector's current limits.
