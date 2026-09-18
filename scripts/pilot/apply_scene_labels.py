"""Apply the model's scene judgments to the deterministic 100-frame pilot."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'pilot' / 'output' / 'pilot_labels.json'

PERSON = {4,5,13,20,21,31,37,38,39,51,60,63,72,78,79,84,94,99,100}
BENCH = {14,43,44,52,80,90}
CROWD = {3,22,50,61,85}
OTHER = {1,2}
MIXED = {42}
OVERHEAD = {10,11,64,65,86,92}
END_CORNER = {8,16,19,24,25,30,32,35,40,46,47,48,54,55,58,66,69,71,73,76,83,87,91,96}
LOW = {49,59,70,77,93}
TIGHT_ACTION = {23,45,57}
PARTIAL = {49,70,93}
UNUSABLE_ACTION = TIGHT_ACTION | {59,77}


def label(index):
    if index in PERSON:
        content, camera, usability = 'person_closeup', 'tight', 'unusable'
    elif index in BENCH:
        content, camera, usability = 'bench_or_penalty_box', 'not_applicable', 'unusable'
    elif index in CROWD:
        content, camera, usability = 'crowd_or_arena', 'not_applicable', 'unusable'
    elif index in OTHER:
        content, camera, usability = 'studio_or_other', 'not_applicable', 'unusable'
    elif index in MIXED:
        content, camera, usability = 'mixed_or_transition', 'not_applicable', 'unusable'
    else:
        content = 'ice_action'
        camera = ('true_overhead' if index in OVERHEAD else
                  'end_or_corner' if index in END_CORNER else
                  'low_rinkside' if index in LOW else
                  'tight' if index in TIGHT_ACTION else 'elevated_wide')
        usability = 'unusable' if index in UNUSABLE_ACTION else 'usable'

    if index in OVERHEAD:
        temporal, temporal_confidence = 'replay', 'medium'
    elif index in {1,2,3,4,5,13,14,21,22}:
        temporal, temporal_confidence = 'unknown', 'low'
    else:
        temporal, temporal_confidence = 'live', 'medium'

    reason = ''
    if content == 'ice_action' and usability == 'unusable':
        reason = 'Tight or low view lacks enough rink geometry to place the visible people reliably.'

    confidence = {
        'content': 'medium' if index in {1,4,23,42,49,50,59,79,93,99,100} else 'high',
        'camera_view': 'medium' if content == 'ice_action' and index in (LOW | TIGHT_ACTION | {83}) else 'high',
        'temporal_context': temporal_confidence,
        'position_usability': 'medium' if index in (PARTIAL | UNUSABLE_ACTION) else 'high',
    }
    review = [key for key, value in confidence.items() if value != 'high']
    return dict(content=content, camera_view=camera, temporal_context=temporal,
                position_usability=usability, reason=reason), confidence, review


def main():
    data = json.loads(PATH.read_text())
    for record in data['records']:
        labels, confidence, review = label(record['pilot_index'])
        record['model_label'] = labels
        record['model_confidence'] = confidence
        record['review_needed'] = review
    PATH.write_text(json.dumps(data, indent=2) + '\n')
    print(f"Applied labels to {len(data['records'])} pilot records")


if __name__ == '__main__':
    main()
