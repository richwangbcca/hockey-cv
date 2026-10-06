# Labeling application

`python3 -m labeling.app` is the stable command-line entry point. The Python
implementation and browser files are in [`src/`](src/README.md). The root
`app.py` keeps existing commands and imports working.

From the repository root on this Mac:

```sh
arch -x86_64 python3 -m labeling.app review-shots --dataset data/datasets/ice_v1
```

Open <http://127.0.0.1:8765> and switch among the four review phases in the
interface. See the [labeling guide](../docs/LABELING_GUIDE.md) for controls,
validation, and export commands.
