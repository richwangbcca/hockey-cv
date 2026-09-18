"""Retrain binary usability with reviewed examples from all three broadcasts."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot import descriptor
from classify_new_broadcasts import SOURCES, ensemble

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pilot" / "output" / "transfer"
INITIAL = OUT / "predictions.json"
REVIEW = ROOT / "pilot" / "transfer_review.json"
SEED = 20260910


def render(source_id, records, prefix):
    directory = SOURCES[source_id]
    for start in range(0, len(records), 12):
        page = np.zeros((3 * 304, 4 * 480, 3), np.uint8)
        for cell, record in enumerate(records[start:start+12]):
            image = cv2.imread(str(directory / record["filename"]))
            image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            row, col = divmod(cell, 4)
            y0, x0 = row * 304, col * 480
            page[y0:y0+270, x0:x0+480] = image
            color = (100, 235, 170) if record["prediction"] == "usable" else (110, 150, 255)
            cv2.putText(page, f'{record["filename"]}  {record["prediction"].upper()}  {record["vote_fraction"]:.2f}',
                        (x0+7, y0+290), cv2.FONT_HERSHEY_SIMPLEX, .46, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{source_id}_{prefix}_{start//12+1:02}.jpg"), page)


def main():
    initial = json.loads(INITIAL.read_text())
    corrections = json.loads(REVIEW.read_text())["corrections"]
    original = json.loads((ROOT / "pilot" / "output" / "pilot_labels.json").read_text())["records"]
    train_x = [descriptor(cv2.imread(str(ROOT / "pool" / record["filename"]))) for record in original]
    train_y = [record["model_label"]["position_usability"] == "usable" for record in original]
    baseline = {}

    for source_id, source in initial["sources"].items():
        by_name = {record["filename"]: record for record in source["records"]}
        reviewed = [record for record in source["records"] if record["transfer_pilot"]]
        errors = 0
        for record in reviewed:
            label = corrections.get(source_id, {}).get(record["filename"], record["prediction"])
            errors += label != record["prediction"]
            path = SOURCES[source_id] / record["filename"]
            train_x.append(descriptor(cv2.imread(str(path))))
            train_y.append(label == "usable")
        baseline[source_id] = {"reviewed": len(reviewed), "correct": len(reviewed)-errors,
                               "accuracy": (len(reviewed)-errors)/len(reviewed)}

    train_x = np.float32(train_x)
    train_y = np.int32(train_y)
    output = {"schema_version": 1, "training_frames": len(train_y), "baseline_transfer": baseline, "sources": {}}
    rng = np.random.default_rng(SEED)
    for source_id, source in initial["sources"].items():
        all_x = np.load(OUT / f"{source_id}_descriptors.npy")
        pred, confidence, distance = ensemble(train_x, train_y, all_x)
        corrections_for_source = corrections.get(source_id, {})
        records = []
        for i, old in enumerate(source["records"]):
            prediction = "usable" if pred[i] else "unusable"
            reviewed_label = None
            if old["transfer_pilot"]:
                reviewed_label = corrections_for_source.get(old["filename"], old["prediction"])
                prediction = reviewed_label
            records.append({**old, "baseline_prediction": old["prediction"],
                            "prediction": prediction, "vote_fraction": float(confidence[i]),
                            "nearest_training_distance": float(distance[i]),
                            "reviewed_label": reviewed_label})
        unreviewed = [r for r in records if not r["transfer_pilot"]]
        uncertain = [r for r in unreviewed if r["vote_fraction"] < 1]
        for record in uncertain:
            record["review_group"] = "model_disagreement"
        audit = []
        for label in ("usable", "unusable"):
            candidates = [r for r in unreviewed if r["vote_fraction"] == 1 and r["prediction"] == label]
            bins = np.array_split(np.arange(len(candidates)), min(15, len(candidates)))
            audit.extend(candidates[int(rng.choice(block))] for block in bins if len(block))
        for record in audit:
            record["review_group"] = "high_confidence_audit"
        render(source_id, uncertain, "uncertain")
        render(source_id, sorted(audit, key=lambda r: r["filename"]), "audit")
        output["sources"][source_id] = {
            "count": len(records), "uncertain_count": len(uncertain), "audit_count": len(audit),
            "predicted_usable": sum(r["prediction"] == "usable" for r in records),
            "predicted_unusable": sum(r["prediction"] == "unusable" for r in records),
            "records": records,
        }
    (OUT / "retrained_predictions.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"training_frames": len(train_y), "baseline_transfer": baseline,
                      "sources": {key: {k: value[k] for k in ("count", "uncertain_count", "audit_count", "predicted_usable", "predicted_unusable")}
                                  for key, value in output["sources"].items()}}))


if __name__ == "__main__":
    main()
