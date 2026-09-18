"""Render model scene labels and a correction CSV for independent review."""
from pathlib import Path
import csv
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / 'pool'
OUT = ROOT / 'pilot' / 'output'

SHORT = {
    'ice_action': 'ICE', 'person_closeup': 'CLOSEUP', 'bench_or_penalty_box': 'BENCH',
    'crowd_or_arena': 'CROWD', 'studio_or_other': 'OTHER', 'mixed_or_transition': 'MIXED',
    'elevated_wide': 'WIDE', 'end_or_corner': 'END/CORNER', 'true_overhead': 'OVERHEAD',
    'low_rinkside': 'LOW', 'tight': 'TIGHT', 'not_applicable': 'N/A',
    'usable': 'USABLE', 'partial': 'PARTIAL', 'unusable': 'UNUSABLE',
}


def main():
    data = json.loads((OUT / 'pilot_labels.json').read_text())
    records = data['records']
    fields = ['content', 'camera_view', 'temporal_context', 'position_usability']
    with (OUT / 'review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pilot_index', 'filename'] +
            [f'model_{x}' for x in fields] + [f'reviewer_{x}' for x in fields] + ['reviewer_notes'])
        writer.writeheader()
        for record in records:
            row = {'pilot_index': record['pilot_index'], 'filename': record['filename'],
                   **{f'model_{x}': record['model_label'][x] for x in fields},
                   **{f'reviewer_{x}': '' for x in fields}, 'reviewer_notes': ''}
            writer.writerow(row)

    for start in range(0, len(records), 12):
        page = np.zeros((3 * 334, 4 * 480, 3), np.uint8)
        for cell, record in enumerate(records[start:start+12]):
            image = cv2.imread(str(POOL / record['filename']))
            thumb = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            row, col = divmod(cell, 4); y, x = row*334, col*480
            page[y:y+270, x:x+480] = thumb
            label = record['model_label']; confidence = record['model_confidence']
            cv2.putText(page, f"{record['pilot_index']:03}  {record['filename']}", (x+7,y+288),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(page, f"{SHORT[label['content']]} | {SHORT[label['camera_view']]} | {SHORT[label['position_usability']]}",
                        (x+7,y+306),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,235,170),1,cv2.LINE_AA)
            low = ', '.join(k.replace('_',' ') for k,v in confidence.items() if v!='high') or 'none'
            cv2.putText(page, f"CHECK: {low}", (x+7,y+324),cv2.FONT_HERSHEY_SIMPLEX,.4,
                        (80,190,255) if low!='none' else (160,160,160),1,cv2.LINE_AA)
        cv2.imwrite(str(OUT / f'labeled_{start//12+1:02}.jpg'), page)
    print(f'Wrote review.csv and {(len(records)+11)//12} labeled contact sheets')


if __name__ == '__main__':
    main()
