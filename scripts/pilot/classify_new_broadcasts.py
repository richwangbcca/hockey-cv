"""Measure and propose binary usability labels on newly added broadcasts."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot import choose, descriptor
from classify_pool import train_svm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pilot" / "output" / "transfer"
SOURCES = {
    "min_vs_sjs": ROOT / "broadcasts" / "min_vs_sjs" / "frames",
    "tor_vs_tbl": ROOT / "broadcasts" / "tor_vs_tbl" / "frames",
}


def ensemble(train_x, y, all_x):
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < .02] = 1
    tx = (train_x - mean) / scale
    x = (all_x - mean) / scale
    specs = [
        (cv2.ml.SVM_LINEAR, .001, None), (cv2.ml.SVM_LINEAR, .01, None),
        (cv2.ml.SVM_RBF, 1, .001), (cv2.ml.SVM_RBF, 10, .0001),
        (cv2.ml.SVM_RBF, 10, .001),
    ]
    votes = [train_svm(tx, y, kernel, c, gamma).predict(x)[1].ravel().astype(int)
             for kernel, c, gamma in specs]
    knn = cv2.ml.KNearest_create()
    knn.train(tx, cv2.ml.ROW_SAMPLE, y.astype(np.float32))
    for k in (3, 5, 7):
        votes.append(knn.findNearest(x, k)[1].ravel().astype(int))
    votes = np.asarray(votes)
    ones = votes.mean(0)
    prediction = (ones >= .5).astype(int)
    confidence = np.where(prediction == 1, ones, 1 - ones)
    distance = (((x[:, None, :] - tx[None, :, :]) ** 2).mean(2) ** .5).min(1)
    return prediction, confidence, distance


def render(source_id, records):
    source = SOURCES[source_id]
    for start in range(0, len(records), 12):
        page = np.zeros((3 * 304, 4 * 480, 3), np.uint8)
        for cell, record in enumerate(records[start:start + 12]):
            image = cv2.imread(str(source / record["filename"]))
            image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            row, col = divmod(cell, 4)
            y0, x0 = row * 304, col * 480
            page[y0:y0+270, x0:x0+480] = image
            color = (100, 235, 170) if record["prediction"] == "usable" else (110, 150, 255)
            text = f'{record["filename"]}  {record["prediction"].upper()}  {record["vote_fraction"]:.2f}'
            cv2.putText(page, text, (x0+7, y0+290), cv2.FONT_HERSHEY_SIMPLEX,
                        .46, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{source_id}_pilot_{start // 12 + 1:02}.jpg"), page)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((ROOT / "pilot" / "output" / "pilot_labels.json").read_text())["records"]
    train_x = np.float32([descriptor(cv2.imread(str(ROOT / "pool" / r["filename"]))) for r in pilot])
    y = np.int32([r["model_label"]["position_usability"] == "usable" for r in pilot])
    payload = {"schema_version": 1, "training_source": "broadcast_01", "training_frames": len(pilot), "sources": {}}

    for source_id, directory in SOURCES.items():
        paths = sorted(directory.glob("*.jpg"))
        cache = OUT / f"{source_id}_descriptors.npy"
        if cache.exists():
            all_x = np.load(cache)
        else:
            all_x = np.float32([descriptor(cv2.imread(str(path))) for path in paths])
            np.save(cache, all_x)
        prediction, confidence, distance = ensemble(train_x, y, all_x)
        selected = set(choose(paths, total=50))
        records = [{
            "filename": path.name,
            "prediction": "usable" if prediction[i] else "unusable",
            "vote_fraction": float(confidence[i]),
            "nearest_pilot_distance": float(distance[i]),
            "transfer_pilot": i in selected,
            "reviewed_label": None,
        } for i, path in enumerate(paths)]
        pilot_records = [r for r in records if r["transfer_pilot"]]
        render(source_id, pilot_records)
        payload["sources"][source_id] = {
            "count": len(records),
            "predicted_usable": int(prediction.sum()),
            "predicted_unusable": int((1 - prediction).sum()),
            "pilot_count": len(pilot_records),
            "records": records,
        }
    (OUT / "predictions.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: {k: v[k] for k in ("count", "predicted_usable", "predicted_unusable", "pilot_count")}
                      for key, v in payload["sources"].items()}))


if __name__ == "__main__":
    main()
