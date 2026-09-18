"""Score pilot predictions after reviewer_* columns in output/review.csv are filled."""
from pathlib import Path
import csv
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'pilot' / 'output' / 'review.csv'
FIELDS = ('content', 'camera_view', 'position_usability', 'temporal_context')


def main():
    rows = list(csv.DictReader(PATH.open()))
    result = {'reviewed_rows': len(rows), 'fields': {}}
    for field in FIELDS:
        pairs = [(r['model_'+field], r['reviewer_'+field].strip()) for r in rows
                 if r['reviewer_'+field].strip()]
        correct = sum(predicted == reviewed for predicted, reviewed in pairs)
        result['fields'][field] = {
            'reviewed': len(pairs),
            'correct': correct,
            'accuracy': correct / len(pairs) if pairs else None,
            'confusion': dict(Counter(f'{predicted} -> {reviewed}' for predicted, reviewed in pairs
                                      if predicted != reviewed)),
        }
    target = PATH.with_name('review_scores.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
