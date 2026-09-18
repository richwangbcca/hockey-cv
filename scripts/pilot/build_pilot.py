"""Build a deterministic, diverse 100-frame scene-labeling pilot from pool/."""
from pathlib import Path
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / 'pool'
OUT = ROOT / 'pilot' / 'output'
SEED = 20260909


def descriptor(image):
    small = cv2.resize(image, (24, 14), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    hist /= max(hist.sum(), 1)
    edges = cv2.Canny(cv2.resize(image, (160, 90)), 80, 160).mean() / 255
    return np.r_[small.ravel(), hist * 4, edges]


def choose(paths, total=100):
    # Ten selections within each temporal tenth preserves broadcast coverage while
    # medoid selection favors different-looking frames over near duplicates.
    selected = []
    for block in np.array_split(np.arange(len(paths)), 10):
        features = np.asarray([descriptor(cv2.imread(str(paths[i]))) for i in block], np.float32)
        features -= features.mean(0)
        scale = features.std(0)
        features /= np.where(scale > .02, scale, 1)
        cv2.setRNGSeed(SEED + int(block[0]))
        _, labels, centers = cv2.kmeans(features, total // 10, None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 80, .01),
            8, cv2.KMEANS_PP_CENTERS)
        for cluster in range(total // 10):
            members = np.flatnonzero(labels.ravel() == cluster)
            distance = ((features[members] - centers[cluster]) ** 2).sum(1)
            selected.append(int(block[members[distance.argmin()]]))
    return sorted(selected)


def main():
    paths = sorted(POOL.glob('*.jpg'))
    OUT.mkdir(parents=True, exist_ok=True)
    indices = choose(paths)
    records = []
    for number, index in enumerate(indices, 1):
        records.append(dict(pilot_index=number, filename=paths[index].name,
            source_index=index, model_label=None, model_confidence=None,
            review_needed=None, reviewer_label=None, reviewer_notes=''))
    (OUT / 'pilot_labels.json').write_text(json.dumps({
        'schema_version': 1,
        'selection': '10 temporal blocks × 10 visual medoids',
        'seed': SEED,
        'source_count': len(paths),
        'records': records,
    }, indent=2) + '\n')

    for start in range(0, len(records), 12):
        page = np.zeros((3 * 294, 4 * 480, 3), np.uint8)
        for cell, record in enumerate(records[start:start + 12]):
            image = cv2.imread(str(POOL / record['filename']))
            thumb = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            row, col = divmod(cell, 4)
            y, x = row * 294, col * 480
            page[y:y+270, x:x+480] = thumb
            cv2.putText(page, f"{record['pilot_index']:03}  {record['filename']}",
                        (x+8, y+288), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f'contact_{start//12+1:02}.jpg'), page)
    print(f'Wrote {len(records)} records and {(len(records)+11)//12} contact sheets to {OUT}')


if __name__ == '__main__':
    main()
