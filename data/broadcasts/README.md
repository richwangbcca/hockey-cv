# Extracted broadcasts

Each broadcast folder contains sparse JPEG frames and a `timing.json` mapping
filenames to source timestamps. Generate a new extraction with
`python3 scripts/prepare_broadcasts.py data/videos/VIDEO.mp4 SOURCE_ID` from
the repository root. Existing extractions are local data and ignored by Git.
