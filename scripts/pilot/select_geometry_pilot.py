"""Select 50 diverse usable frames per broadcast for the geometry pilot."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from build_pilot import descriptor
from labeling.app import atomic_json

DATASET = ROOT / "datasets" / "ice_v1"
OUT = DATASET / "selections"
NAME = "geometry_pilot_v1"
SOURCE_IDS = ("broadcast_01", "min_vs_sjs", "tor_vs_tbl")
SEED = 20260910


def select(entries, features, count=50):
    chosen = []
    per_block = count // 10
    for block_number, block in enumerate(np.array_split(np.arange(len(entries)), 10)):
        x = features[block].copy()
        x -= x.mean(0)
        scale = x.std(0)
        x /= np.where(scale > .02, scale, 1)
        cv2.setRNGSeed(SEED + block_number + len(entries))
        _, labels, centers = cv2.kmeans(
            x, per_block, None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 80, .01),
            8, cv2.KMEANS_PP_CENTERS,
        )
        for cluster in range(per_block):
            members = np.flatnonzero(labels.ravel() == cluster)
            distances = ((x[members] - centers[cluster]) ** 2).sum(1)
            chosen.append(int(block[members[distances.argmin()]]))
    return sorted(chosen)


def render(source_id, selected):
    for start in range(0, len(selected), 12):
        page = np.zeros((3 * 300, 4 * 480, 3), np.uint8)
        for cell, item in enumerate(selected[start:start+12]):
            image = cv2.imread(item["path"])
            image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            row, col = divmod(cell, 4)
            y0, x0 = row * 300, col * 480
            page[y0:y0+270, x0:x0+480] = image
            stamp = "timing unknown" if item["timestamp_seconds"] is None else f'{item["timestamp_seconds"]:.0f}s'
            cv2.putText(page, f'{item["relative_path"]}  {stamp}', (x0+7, y0+290),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{NAME}_{source_id}_{start//12+1:02}.jpg"), page)


def main():
    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    selected_all = []
    source_counts = {}
    for source_id in SOURCE_IDS:
        entries = [entry for entry in manifest["images"] if entry["source_id"] == source_id]
        entries.sort(key=lambda entry: (entry["timestamp_seconds"] is None,
                                        entry["timestamp_seconds"] or 0, entry["relative_path"]))
        usable = []
        for entry in entries:
            record = json.loads((DATASET / "records" / f'{entry["id"]}.json').read_text())
            if record["review"]["shot"] == "reviewed" and record["shot"]["position_usability"] == "usable":
                usable.append(entry)
        if len(usable) < 50:
            raise ValueError(f"{source_id} has only {len(usable)} usable frames")
        features = np.float32([descriptor(cv2.imread(entry["path"])) for entry in usable])
        selected = [usable[index] for index in select(usable, features)]
        serialized = [{key: entry.get(key) for key in (
            "id", "source_id", "relative_path", "path", "timestamp_seconds", "source_frame_index"
        )} for entry in selected]
        selected_all.extend(serialized)
        source_counts[source_id] = len(serialized)
        render(source_id, serialized)

    payload = {
        "schema_version": 1,
        "name": NAME,
        "criterion": "reviewed usable frames; 10 temporal blocks x 5 visual medoids per broadcast",
        "seed": SEED,
        "count": len(selected_all),
        "source_counts": source_counts,
        "images": selected_all,
    }
    atomic_json(OUT / f"{NAME}.json", payload)
    manifest.setdefault("selections", {})[NAME] = [item["id"] for item in selected_all]
    atomic_json(manifest_path, manifest)
    print(json.dumps({"selection": NAME, "count": len(selected_all), "source_counts": source_counts}))


if __name__ == "__main__":
    main()
