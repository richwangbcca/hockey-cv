# Tests

Run from the repository root:

```sh
arch -x86_64 python3 -m unittest discover -s tests -v
node --check labeling/src/web/app.js
```

`browser_fixture.py` builds a disposable dataset from `data/frames/601.png`.
`browser_smoke.mjs` exercises the browser against that fixture; its setup is
documented in the [labeling guide](../docs/LABELING_GUIDE.md).
